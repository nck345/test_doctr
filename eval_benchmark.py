import os
import argparse
import time
import cv2
import csv
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from rapidocr_onnxruntime import RapidOCR

try:
	from rapidfuzz.distance import Levenshtein
	def get_levenshtein(s1,s2):
		return Levenshtein.distance(s1,s2)
except ImportError:
	def get_levenshtein(s1,s2):
		matrix=[[i+j for j in range(len(s2)+1)] for i in range(len(s1)+1)]
		for i in range(1,len(s1)+1):
			for j in range(1,len(s2)+1):
				d=0 if s1[i-1]==s2[j-1] else 1
				matrix[i][j]=min(matrix[i-1][j]+1,matrix[i][j-1]+1,matrix[i-1][j-1]+d)
		return matrix[len(s1)][len(s2)]

def extract_doc_id(filename):
	base=os.path.basename(filename)
	prefix=base.split('_')[0]
	prefix=''.join(filter(str.isdigit,prefix))
	return prefix

def main():
	parser=argparse.ArgumentParser(description="Universal Evaluation Toolkit for Document Image Dewarping Models")
	parser.add_argument('--pred_dir','--rec_dir',dest='pred_dir',default='./results',help='Directory containing rectified/dewarped images from ANY model')
	parser.add_argument('--gt_dir','--scan_dir',dest='gt_dir',default='./data benchmark/scan/scan',help='Directory containing ground truth scan images')
	parser.add_argument('--model_name',default='Model',help='Name of the evaluated model (e.g. DocTr, DewarpNet, Custom)')
	parser.add_argument('--save_csv',default='',help='Optional path to save detailed evaluation report as CSV')
	args=parser.parse_args()
	if not os.path.exists(args.pred_dir):
		print(f"Error: Prediction directory '{args.pred_dir}' does not exist.")
		return
	if not os.path.exists(args.gt_dir):
		print(f"Error: Ground truth directory '{args.gt_dir}' does not exist.")
		return
	ocr=RapidOCR()
	pred_files=[f for f in os.listdir(args.pred_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
	pred_files.sort()
	total_files=len(pred_files)
	if total_files==0:
		print(f"No valid images found in '{args.pred_dir}'.")
		return
	print(f"=== Universal Benchmark Evaluation ===")
	print(f"Model Name    : {args.model_name}")
	print(f"Predictions   : {args.pred_dir} ({total_files} images)")
	print(f"Ground Truth  : {args.gt_dir}")
	print("="*95)
	print(f"{'Index':<6} | {'Dewarped Image':<25} | {'Scan GT':<10} | {'SSIM':<8} | {'CER':<8} | {'ED':<6} | {'Time (s)':<8}")
	print("-" * 95)
	ssim_list=[]
	cer_list=[]
	ed_list=[]
	time_list=[]
	detailed_records=[]
	evaluated_count=0
	for idx,fname in enumerate(pred_files,1):
		doc_id=extract_doc_id(fname)
		scan_name=f"{doc_id}.png"
		scan_path=os.path.join(args.gt_dir,scan_name)
		if not os.path.exists(scan_path):
			continue
		pred_path=os.path.join(args.pred_dir,fname)
		t_start=time.time()
		pred_bgr=cv2.imread(pred_path)
		scan_bgr=cv2.imread(scan_path)
		if pred_bgr is None or scan_bgr is None:
			continue
		pred_gray=cv2.cvtColor(pred_bgr,cv2.COLOR_BGR2GRAY)
		scan_gray=cv2.cvtColor(scan_bgr,cv2.COLOR_BGR2GRAY)
		pred_gray_resized=cv2.resize(pred_gray,(scan_gray.shape[1],scan_gray.shape[0]))
		cur_ssim=float(ssim(scan_gray,pred_gray_resized))
		pred_rgb=cv2.cvtColor(pred_bgr,cv2.COLOR_BGR2RGB)
		scan_rgb=cv2.cvtColor(scan_bgr,cv2.COLOR_BGR2RGB)
		res_pred,_=ocr(pred_rgb)
		res_scan,_=ocr(scan_rgb)
		text_pred=" ".join([line[1] for line in res_pred]) if res_pred else ""
		text_scan=" ".join([line[1] for line in res_scan]) if res_scan else ""
		cur_ed=get_levenshtein(text_scan,text_pred)
		cur_cer=float(cur_ed/max(1,len(text_scan)))
		cur_time=time.time()-t_start
		ssim_list.append(cur_ssim)
		cer_list.append(cur_cer)
		ed_list.append(cur_ed)
		time_list.append(cur_time)
		evaluated_count+=1
		detailed_records.append([idx,args.model_name,fname,scan_name,f"{cur_ssim:.4f}",f"{cur_cer:.4f}",cur_ed,f"{cur_time:.2f}"])
		print(f"{idx:<6} | {fname:<25} | {scan_name:<10} | {cur_ssim:<8.4f} | {cur_cer:<8.4f} | {cur_ed:<6} | {cur_time:<8.2f}")
	if evaluated_count>0:
		mean_ssim=np.mean(ssim_list)
		mean_cer=np.mean(cer_list)
		mean_ed=np.mean(ed_list)
		mean_time=np.mean(time_list)
		print("="*95)
		print(f"KET QUA DANH GIA TONG THE CHO [{args.model_name.upper()}]:")
		print(f"- So luong anh hop le          : {evaluated_count} anh")
		print(f"- MS-SSIM trung binh (↑)       : {mean_ssim:.4f}")
		print(f"- CER trung binh (↓)           : {mean_cer:.4f} ({mean_cer*100:.2f}%)")
		print(f"- Levenshtein ED trung binh (↓): {mean_ed:.2f} ky tu/trang")
		print(f"- Thoi gian danh gia TB        : {mean_time:.2f} s/anh")
		print("="*95)
		if args.save_csv:
			csv_dir=os.path.dirname(args.save_csv)
			if csv_dir and not os.path.exists(csv_dir):
				os.makedirs(csv_dir,exist_ok=True)
			with open(args.save_csv,'w',newline='',encoding='utf-8') as f:
				writer=csv.writer(f)
				writer.writerow(['Index','Model','Dewarped_Image','Scan_GT','SSIM','CER','ED','Time_seconds'])
				writer.writerows(detailed_records)
				writer.writerow([])
				writer.writerow(['SUMMARY',args.model_name,f"Total: {evaluated_count}",'All',f"{mean_ssim:.4f}",f"{mean_cer:.4f}",f"{mean_ed:.2f}",f"{mean_time:.2f}"])
			print(f"Report saved to CSV: {args.save_csv}")

if __name__=='__main__':
	main()
