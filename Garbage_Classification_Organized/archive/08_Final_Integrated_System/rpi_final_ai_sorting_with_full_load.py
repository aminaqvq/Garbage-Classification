#!/usr/bin/env python3
"""★ 最终树莓派上位机：超声波触发 + AI + 满载保护 + ASCII RKHO"""
import argparse, csv, json, logging, os, sys, time, traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
os.environ.setdefault("QT_QPA_PLATFORM","xcb")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
import cv2, numpy as np, serial
from PIL import Image, ImageDraw, ImageFont
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf; Interpreter = tf.lite.Interpreter

MCU_TRIGGER_CHAR="T"; MCU_FULL_CHAR="F"; MCU_NORMAL_CHAR="N"; MCU_DONE_CHAR="D"; MCU_ERROR_CHAR="E"
VALID_RX={"T","F","N","D","E"}
CLASS_TO_MCU={"可回收":"R","厨余":"K","有害":"H","其他":"O"}
DEFAULT_IDX={0:"其他",1:"厨余",2:"可回收",3:"有害"}
DISPLAY={"可回收":"可回收垃圾","厨余":"厨余垃圾","有害":"有害垃圾","其他":"其他垃圾"}

PROJECT_ROOT=Path(__file__).resolve().parent.parent.parent
DMODEL=PROJECT_ROOT/"export"/"latest_tflite_fp16.tflite"
DMAP=PROJECT_ROOT/"config"/"class_mapping.json"
DLOG=PROJECT_ROOT/"Logs"; DCAP=PROJECT_ROOT/"Captures_Final"
FONTS=["/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc","C:/Windows/Fonts/msyh.ttc"]

def ns(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def ts(): return datetime.now().strftime("%Y%m%d_%H%M%S")
def hs(d): return " ".join(f"{b:02X}" for b in d)
def ensure(d): d.mkdir(parents=True,exist_ok=True)

class FinalSortingApp:
    def __init__(self, args):
        self.a=args; self.logger=self._setup_logger()
        self.logger.info("最终上位机 — ASCII RKHO + T/F/N/D/E")
        idx=self._load_map(); self.clf=GarbageClassifier(self.a.model_path or str(DMODEL),idx,self.logger)
        self.stable=StablePredictor(args.conf_threshold, args.stable_frames)
        self.ser=SerialMgr(args.serial_port, args.baudrate, self.logger)
        self.csv=DLOG/"final_rounds.csv"; ensure(DLOG,DCAP)
        self.rid=0; self.run=True; self.fps=0.0; self.lf=time.time()
        self.state="WAIT_TRIGGER"; self.full=False; self.sts="等待MCU触发T"
        self.ct="-"; self.cft="-"; self.sbt="-"; self.srt="未通信"; self.rx=""; self.show_frame=None
        self._cap=None

    def _setup_logger(self):
        ensure(DLOG); l=logging.getLogger("Final"); l.setLevel(logging.INFO); l.handlers.clear()
        f=logging.Formatter("[%(asctime)s] %(message)s","%H:%M:%S")
        h=logging.StreamHandler(sys.stdout); h.setFormatter(f); l.addHandler(h)
        return l

    def _load_map(self):
        p=Path(self.a.class_mapping or str(DMAP))
        if p.exists():
            try:
                d=json.loads(p.read_text(encoding="utf-8"))
                for k in ("idx_to_class","class_to_idx"):
                    m=d.get(k)
                    if isinstance(m,dict): return {int(a):str(b) for a,b in m.items()} if k=="idx_to_class" else {int(b):str(a) for a,b in m.items()}
            except: pass
        return DEFAULT_IDX.copy()

    def _append_csv(self, row):
        p=self.csv; ensure(p.parent); ex=p.exists()
        flds=["time","round_id","event","pred_id","raw_class","display_class","confidence","stable_count","tx_char","tx_hex","rx_hex","mcu_event","system_state","message"]
        with open(p,"a",newline="",encoding="utf-8-sig") as f:
            w=csv.DictWriter(f,fieldnames=flds)
            if not ex: w.writeheader()
            w.writerow({k:row.get(k,"") for k in flds})

    def ss(self): return "FULL" if self.full else self.state

    def process_mcu_events(self):
        d=self.ser.read_available()
        if not d: return False
        imp=False
        for b in d:
            ch=chr(b)
            if ch not in VALID_RX: continue
            self.rx=f"RX '{ch}'/0x{b:02X}"; self.srt=self.rx
            if ch=="F":
                self.full=True; self.state="FULL"; self.sts="垃圾桶已满,请清理"; self.stable.reset()
                self.logger.warning("MCU F — 满载暂停")
                self._append_csv({"time":ns(),"round_id":self.rid,"event":"mcu_full","rx_hex":hs(bytes([b])),"mcu_event":"FULL","system_state":"FULL","message":"满载暂停"}); imp=True
            elif ch=="N":
                self.full=False; self.state="WAIT_TRIGGER"; self.sts="满载解除,等待触发T"
                self.logger.info("MCU N — 恢复")
                self._append_csv({"time":ns(),"round_id":self.rid,"event":"mcu_normal","rx_hex":hs(bytes([b])),"mcu_event":"NORMAL","system_state":"NORMAL","message":"满载解除"}); imp=True
            elif ch=="T" and not self.full:
                self.state="PREDICT"; self.sts="收到T,识别中"; self.stable.reset(); self.rid+=1
                self.logger.info("MCU T — round %d",self.rid); imp=True
            elif ch=="D":
                self.state="WAIT_TRIGGER"; self.sts="分拣完成D"
                self._append_csv({"time":ns(),"round_id":self.rid,"event":"mcu_done","rx_hex":hs(bytes([b])),"mcu_event":"DONE","system_state":self.ss(),"message":"分拣完成"}); imp=True
            elif ch=="E":
                self.state="WAIT_TRIGGER"; self.sts="MCU错误E"
                self._append_csv({"time":ns(),"round_id":self.rid,"event":"mcu_error","rx_hex":hs(bytes([b])),"mcu_event":"ERROR","system_state":self.ss(),"message":"MCU错误"}); imp=True
        return imp

    def send_class(self, result):
        raw=result["raw_class"]; ch=CLASS_TO_MCU.get(raw)
        if not ch: self.logger.error("无法映射 %s",raw); return
        if self.full: self.logger.warning("满载禁止发送"); return
        try:
            tx=self.ser.send_char(ch); self.srt=f"TX '{ch}'/0x{tx[0]:02X}"
            self.sts=f"已发送{ch}"; self.state="WAIT_DONE"
            snap=""
            if self.a.save_on_send and self.show_frame is not None:
                fp=DCAP/f"r{self.rid:04d}_{raw}_{result['confidence']:.3f}_{ts()}.jpg"
                cv2.imwrite(str(fp),self.show_frame); snap=str(fp)
            self._append_csv({"time":ns(),"round_id":self.rid,"event":"send","pred_id":result["pred_id"],"raw_class":raw,"display_class":result["display_class"],"confidence":f"{result['confidence']:.6f}","tx_char":ch,"tx_hex":hs(tx),"system_state":self.ss(),"snapshot_path":snap,"message":"已发送"})
        except Exception as e: self.logger.exception("发送失败: %s",e)

    def draw(self, frame, roi, color=(0,255,0)):
        x1,y1,x2,y2=roi; s=frame.copy(); cv2.rectangle(s,(x1,y1),(x2,y2),color,2)
        o=s.copy(); cv2.rectangle(o,(10,10),(630,310 if self.full else 280),(0,0,0),-1)
        s=cv2.addWeighted(o,0.45,s,0.55,0)
        ls=[f"状态:{self.sts}",f"类别:{self.ct}",f"置信度:{self.cft}",f"稳定帧:{self.sbt}",f"串口:{self.srt}",f"模式:{self.state} FPS:{self.fps:.1f}",f"轮次:{self.rid}"]
        if self.full: ls.append("按键: q退出 s截图")
        else: ls.append("按键: q退出 s截图 r/k/h/o手动测试")
        y=34
        for l in ls:
            sl=l.encode("ascii",errors="replace").decode("ascii")
            cv2.putText(s,sl,(22,y),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,0),2); y+=30
        cv2.putText(s,"ROI",(x1+8,y1-8 if y1>25 else y1+25),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)
        return s

    def run(self):
        self.ser.open(); self._open_cam()
        if not self.a.no_window: cv2.namedWindow("FinalSort",cv2.WINDOW_NORMAL)
        try:
            while self.run:
                self.process_mcu_events()
                f=self._read_frame(); n=time.time(); dt=n-self.lf; self.lf=n
                if dt>0: self.fps=0.9*self.fps+0.1*(1.0/dt) if self.fps>0 else 1.0/dt
                h,w=f.shape[:2]; rw,rh=int(w*.56),int(h*.70)
                x1,y1=max(0,(w-rw)//2),max(0,(h-rh)//2)
                x2,y2=min(x1+rw,w),min(y1+rh,h)
                roi_bgr=f[y1:y2,x1:x2]; rb=(x1,y1,x2,y2)
                if self.full or self.state in ("WAIT_TRIGGER","WAIT_DONE"):
                    s=self.draw(f,rb); self.show_frame=s
                elif self.state=="PREDICT":
                    r=self.clf.predict(roi_bgr); si=self.stable.update(r)
                    self.ct=r["display_class"]; self.cft=f"{r['confidence']:.3f}"
                    self.sbt=str(si["stable_count"]); self.sts=si["status"]
                    c=(0,255,0) if si["is_stable"] else (255,255,0)
                    s=self.draw(f,rb,c); self.show_f
