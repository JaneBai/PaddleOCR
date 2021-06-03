python tools/infer/predict_system.py --image_dir="E:/image/OCR/FangZheng/0107/OCR" --det_model_dir="./inference/ch_ppocr_mobile_v1.1_det_infer/"  --rec_model_dir="./inference/ch_ppocr_mobile_v1.1_rec_infer/" --cls_model_dir="./inference/ch_ppocr_mobile_v1.1_cls_infer/" --use_angle_cls=True --use_space_char=True

python tools/infer/predict_system.py --image_dir="F:\\image\\OCR\\FangZheng\\err" --det_model_dir="./inference/ch_ppocr_mobile_v2.0_det_infer/"  --rec_model_dir="./inference/ch_ppocr_mobile_v2.0_rec_infer/" --cls_model_dir="./inference/ch_ppocr_mobile_v2.0_cls_infer/" --use_angle_cls=True --use_space_char=True

python tools/infer/predict_system.py --image_dir="F:\\image\\OCR\\FangZheng\\err" --det_model_dir="./inference/ch_ppocr_server_v2.0_det_infer/"  --rec_model_dir="./inference/ch_ppocr_server_v2.0_rec_infer/" --cls_model_dir="./inference/ch_ppocr_mobile_v2.0_cls_infer/" --use_angle_cls=True --use_space_char=True

python tools/infer/predict_det.py --image_dir="F:\\image\\OCR\\FangZheng\\shumaguan" --det_model_dir="./inference/ch_ppocr_server_v2.0_det_infer/"  --use_space_char=True

//Step1:训练检测模型，保存在“Global.save_model_dir”下。默认：./output/det_r50_vd

python tools/train.py -c configs/det/det_r50_vd_db.yml -o Global.pretrain_weights=./pretrain_models/ResNet50_vd_ssld_pretrained/ 

//Step2:评估检测模型,保存在“Global.save_res_path”下。默认：./output/det_db/predicts_db.txt
#注意：后面设置box_thresh和unclip_ratio是为了将ppocr\postprocess\db_postprocess.py中的参数跟det_r50_vd_db.yml中的后处理参数PostProcess的参数统一
python tools/eval.py -c configs/det/det_r50_vd_db.yml  -o Global.checkpoints="./output/det_r50_vd/best_accuracy" PostProcess.box_thresh=0.7 PostProcess.unclip_ratio=1.5

//Step3:测试图像
python tools/infer_det.py -c configs/det/det_r50_vd_db.yml -o Global.infer_img="D:/4_code/GitHub_Open/PaddleOCR/train_data/text_localization/train" Global.pretrained_model="./output/det_r50_vd/best_accuracy" Global.load_static_weights=false PostProcess.box_thresh=0.7 PostProcess.unclip_ratio=1.5

//训练识别模型
python tools/train.py -c configs/rec/rec_icdar15_train.yml
//转inference模型
python tools/export_model.py -c configs/rec/rec_icdar15_train.yml -o Global.pretrained_model=output/rec/ic15/best_accuracy Global.load_static_weights=False Global.save_inference_dir=./inference/rec_IC15/
//??
python tools/infer/predict_rec.py --image_dir="D:\\4_code\\GitHub_Open\\PaddleOCR\\train_data\\eval" --det_model_dir="./inference/ch_ppocr_mobile_v2.0_det_infer/"  --rec_model_dir="./inference/rec_IC15/" --cls_model_dir="./inference/ch_ppocr_mobile_v2.0_cls_infer/" --use_angle_cls=True --use_space_char=True
python tools/eval.py -c configs/rec/rec_icdar15_train.yml -o Global.checkpoints=output/rec/ic15/best_accuracy

//运行pplabel
python PPOCRLabel.py --lang ch

//数据合成
python tools/synth_image.py -c configs/config.yml --style_image examples/style_images/2.png --text_corpus PaddleOCR --language en

