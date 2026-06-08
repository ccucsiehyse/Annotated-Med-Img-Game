import os
import sys
import glob
import random
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import open_clip
from google import genai

# ==========================================
# 🔑 請在這裡貼上你的 Google Gemini API Key
GEMINI_API_KEY = "YOUR_API_KEY_HERE" 
# ==========================================

# 加入 MedSAM 路徑
sys.path.append(os.path.abspath('./MedSAM'))
from segment_anything import sam_model_registry, SamPredictor

def compute_dice(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    total = pred_mask.sum() + gt_mask.sum()
    return 1.0 if total == 0 and intersection == 0 else (2.0 * intersection) / (total + 1e-8)

def create_spatial_prior_mask(shape=(14, 14), sigma=3.0):
    x = np.linspace(-1, 1, shape[1])
    y = np.linspace(-1, 1, shape[0])
    x, y = np.meshgrid(x, y)
    d = np.sqrt(x*x + y*y)
    gaussian_mask = np.exp(-(d**2 / (2.0 * sigma**2)))
    gaussian_mask = (gaussian_mask - gaussian_mask.min()) / (gaussian_mask.max() - gaussian_mask.min() + 1e-8)
    return gaussian_mask

def draw_chinese_text(img_bgr, text, position, font_size=30, text_color=(0, 255, 255)):
    """【黑科技】解決 OpenCV 無法顯示中文的問題，使用 PIL 繪製微軟正黑體"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    try:
        # 嘗試載入 Windows 內建的微軟正黑體
        font = ImageFont.truetype("msjh.ttc", font_size)
    except IOError:
        print("⚠️ 找不到微軟正黑體，切換至預設字體")
        font = ImageFont.load_default()
    
    # 畫一點黑色陰影讓文字更清楚
    x, y = position
    draw.text((x+2, y+2), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=text_color)
    
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def get_gemini_feedback(image_bgr, score):
    """【API 串接】將結算長圖傳給 Gemini API 進行動態點評"""
    if GEMINI_API_KEY == "YOUR_API_KEY_HERE":
        return "⚠️ 請先在程式碼中填入 Gemini API Key 才能解鎖 AI 點評功能！"
        
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # 將 OpenCV BGR 圖片轉為 PIL RGB 圖片給 API 讀取
        pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        
        # 魔法提示詞 (Prompt Engineering)
        prompt = f"""
        你是一位嚴格但幽默的神經外科主治醫師。你正在指導醫學生標註腦出血 CT。
        這張長圖左邊是學生畫的標註框，中間是 AI 切割結果，右邊是真實解答。
        系統根據重疊度給出了學生的分數為：{score} 分 (滿分100)。
        請觀察學生標註的或AI(MedSAM)切割的結果如何？
        請用 1 句簡短的繁體中文（不超過 30 個字）直接給予他神經外科醫師口吻的點評。
        """
        # 更新至最新的 Gemini 2.5 Flash 模型 (若仍報錯可改為 'gemini-2.0-flash')
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, pil_img]
        )
        return response.text.strip().replace('\n', ' ')
    except Exception as e:
        print(f"Gemini API 發生錯誤: {e}")
        return "通訊干擾，主治醫師暫時無法連線。"

class MedAIGame:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🎮 歡迎來到 MedAI 標註挑戰賽！(使用裝置: {self.device})")
        
        print("📥 正在載入 VLM 導師 (BiomedCLIP)...")
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.clip_model.to(self.device).eval()

        print("📥 正在載入 像素裁判 (MedSAM)...")
        medsam_model = sam_model_registry["vit_b"](checkpoint="./medsam_vit_b.pth")
        medsam_model.to(self.device).eval()
        self.predictor = SamPredictor(medsam_model)

        print("✅ 遊戲引擎啟動完成！")

        dataset_dir = "./computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0/Patients_CT"
        self.level_files = []
        for pid in os.listdir(dataset_dir):
            brain_dir = os.path.join(dataset_dir, pid, "brain")
            if os.path.exists(brain_dir):
                mask_paths = glob.glob(os.path.join(brain_dir, "*_HGE_Seg.jpg"))
                for mp in mask_paths:
                    img_p = mp.replace("_HGE_Seg.jpg", ".jpg")
                    if os.path.exists(img_p):
                        self.level_files.append((img_p, mp))
        
        random.shuffle(self.level_files)
        self.current_level = -1
        self.window_name = "MedAI Game - Annotation Challenge"
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.state = "PLAYING"
        self.drawing = False
        self.show_hint = False
        self.ix, self.iy = -1, -1
        self.cx, self.cy = -1, -1
        self.bbox = None
        self.result_screen = None

        self.load_next_level()

    def load_next_level(self):
        self.current_level += 1
        if self.current_level >= len(self.level_files):
            print("🎉 恭喜你通關了所有題庫！")
            sys.exit()

        img_path, mask_path = self.level_files[self.current_level]
        self.img_bgr = cv2.imread(img_path)
        self.img_bgr = cv2.resize(self.img_bgr, (600, 600))
        self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (600, 600), interpolation=cv2.INTER_NEAREST)
        _, self.gt_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

        self.generate_vlm_hint()
        self.reset_level_state()

    def reset_level_state(self):
        self.state = "PLAYING"
        self.drawing = False
        self.show_hint = False
        self.bbox = None
        self.ix, self.iy, self.cx, self.cy = -1, -1, -1, -1

    def generate_vlm_hint(self):
        pil_image = Image.fromarray(self.img_rgb)
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        text_input = self.tokenizer(["hyperdense intracranial hemorrhage lesion", "normal healthy brain tissue, skull bone, background"]).to(self.device)

        with torch.no_grad(), torch.amp.autocast('cuda'):
            text_features = self.clip_model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            image_features_all = self.clip_model.visual.trunk.forward_features(image_input)
            patch_features = image_features_all[:, 1:, :] 
            patch_features = self.clip_model.visual.head.proj(patch_features)
            patch_features /= patch_features.norm(dim=-1, keepdim=True)

            logits = 100.0 * patch_features @ text_features.T
            prob_hemorrhage = logits.softmax(dim=-1).squeeze()[:, 0]
            
            heatmap = prob_hemorrhage.reshape(14, 14).cpu().numpy()
            heatmap = heatmap * create_spatial_prior_mask()
            heatmap = heatmap.astype(np.float32)
            
            heatmap_resized = cv2.resize(heatmap, (600, 600), interpolation=cv2.INTER_CUBIC)
            heatmap_resized = cv2.GaussianBlur(heatmap_resized, (15, 15), 0)
            
            heatmap_vis = np.uint8(255 * heatmap_resized)
            self.heatmap_color = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)

    def mouse_callback(self, event, x, y, flags, param):
        if self.state != "PLAYING": return

        if event == cv2.EVENT_LBUTTONDOWN:
            if x >= 600: return 
            self.drawing = True
            self.ix, self.iy = x, y
            self.cx, self.cy = x, y
            self.bbox = None

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.cx = min(x, 599)
                self.cy = min(y, 599)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                self.cx = min(x, 599)
                self.cy = min(y, 599)
                
                xmin, xmax = min(self.ix, self.cx), max(self.ix, self.cx)
                ymin, ymax = min(self.iy, self.cy), max(self.iy, self.cy)
                
                if xmax - xmin > 5 and ymax - ymin > 5:
                    self.bbox = np.array([xmin, ymin, xmax, ymax])
                else:
                    self.bbox = None

    def process_submission(self):
        if self.bbox is None:
            return

        self.state = "EVALUATING" # 鎖定畫面避免干擾

        print("🤖 提交給 MedSAM 裁判計算中...")
        self.predictor.set_image(self.img_rgb)
        masks, _, _ = self.predictor.predict(box=self.bbox, multimask_output=False)
        pred_mask = masks[0].astype(np.uint8)

        dice = compute_dice(pred_mask, self.gt_mask)
        score = int(dice * 100)

        # 構建結算長圖
        self.result_screen = np.zeros((600, 1200, 3), dtype=np.uint8)

        vis_user = self.img_bgr.copy()
        cv2.rectangle(vis_user, (self.bbox[0], self.bbox[1]), (self.bbox[2], self.bbox[3]), (255, 0, 0), 2)
        vis_user_resized = cv2.resize(vis_user, (400, 400))

        vis_pred = cv2.cvtColor(pred_mask * 255, cv2.COLOR_GRAY2BGR)
        vis_pred_resized = cv2.resize(vis_pred, (400, 400))

        vis_gt = cv2.cvtColor(self.gt_mask * 255, cv2.COLOR_GRAY2BGR)
        vis_gt_resized = cv2.resize(vis_gt, (400, 400))

        self.result_screen[0:400, 0:400] = vis_user_resized
        self.result_screen[0:400, 400:800] = vis_pred_resized
        self.result_screen[0:400, 800:1200] = vis_gt_resized
        
        cv2.rectangle(self.result_screen, (10, 5), (140, 35), (0,0,0), -1)
        cv2.putText(self.result_screen, "1. Your BBox", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
        cv2.rectangle(self.result_screen, (410, 5), (550, 35), (0,0,0), -1)
        cv2.putText(self.result_screen, "2. MedSAM Cut", (415, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.rectangle(self.result_screen, (810, 5), (960, 35), (0,0,0), -1)
        cv2.putText(self.result_screen, "3. True Answer", (815, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

        if score >= 80:
            color = (0, 255, 0)
        elif score >= 50:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

        cv2.putText(self.result_screen, f"SCORE: {score}", (40, 480), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)
        
        # --- UI 魔法：先顯示等待中的畫面 ---
        wait_screen = self.result_screen.copy()
        wait_screen = draw_chinese_text(wait_screen, "🤖 等待主治醫師 (Gemini) 讀片點評中...", (40, 520), font_size=30, text_color=(200, 200, 200))
        cv2.imshow(self.window_name, wait_screen)
        cv2.waitKey(1) # 強制更新視窗畫面

        # --- 呼叫 Gemini API 進行真實的看圖推論 ---
        print("🧠 正在呼叫 Gemini 主治醫師分析畫面佈局...")
        # 我們只切出上半部的圖片 (不包含分數文字)，讓 AI 完全靠圖片+我們給的分數提示來思考
        api_img = self.result_screen[0:400, 0:1200] 
        gemini_comment = get_gemini_feedback(api_img, score)
        print(f"醫師點評: {gemini_comment}")

        # --- 把 AI 的繁體中文評語畫到最終畫面上 ---
        self.result_screen = draw_chinese_text(self.result_screen, f"主治醫師點評：{gemini_comment}", (40, 520), font_size=28, text_color=(0, 255, 255))
        cv2.putText(self.result_screen, "[ N ] Next Level   |   [ R ] Retry Level   |   [ Q ] Quit", (650, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        self.state = "RESULT"

    def run(self):
        print("\n==============================================")
        print(" 🕹️ 遊戲已在獨立視窗開啟！請查看視窗。")
        print("==============================================\n")

        font = cv2.FONT_HERSHEY_SIMPLEX

        while True:
            canvas = np.zeros((600, 1200, 3), dtype=np.uint8)

            if self.state == "PLAYING":
                display_img = self.img_bgr.copy()

                if self.show_hint:
                    display_img = cv2.addWeighted(display_img, 0.5, self.heatmap_color, 0.5, 0)

                if self.drawing:
                    cv2.rectangle(display_img, (self.ix, self.iy), (self.cx, self.cy), (0, 165, 255), 2)
                elif self.bbox is not None:
                    cv2.rectangle(display_img, (self.bbox[0], self.bbox[1]), (self.bbox[2], self.bbox[3]), (255, 0, 0), 2)

                canvas[0:600, 0:600] = display_img

                cv2.putText(canvas, "MedAI Annotation Challenge", (630, 60), font, 1.0, (255, 255, 255), 2)
                cv2.putText(canvas, "-"*35, (630, 90), font, 0.8, (150, 150, 150), 1)
                
                cv2.putText(canvas, "[Mouse Drag] : Draw BBox", (630, 150), font, 0.8, (200, 255, 200), 2)
                cv2.putText(canvas, "[ H ] : Toggle BiomedCLIP Radar", (630, 200), font, 0.8, (200, 255, 200), 2)
                cv2.putText(canvas, "[ R ] : Clear BBox", (630, 250), font, 0.8, (200, 255, 200), 2)
                cv2.putText(canvas, "[Enter] : Submit to MedSAM & Gemini", (630, 300), font, 0.8, (200, 255, 200), 2)
                cv2.putText(canvas, "[ Q ] : Quit Game", (630, 350), font, 0.8, (200, 255, 200), 2)

                if self.show_hint:
                    cv2.putText(canvas, ">> VLM RADAR: ON <<", (630, 480), font, 1.0, (0, 0, 255), 2)
                if self.bbox is not None:
                    cv2.putText(canvas, "BBox Ready! Press [ENTER]", (630, 540), font, 0.8, (255, 165, 0), 2)

                cv2.imshow(self.window_name, canvas)

            elif self.state == "RESULT":
                cv2.imshow(self.window_name, self.result_screen)

            key = cv2.waitKey(15) & 0xFF
            
            if key == ord('q') or key == 27: 
                break
            
            if self.state == "PLAYING":
                if key == ord('h'):
                    self.show_hint = not self.show_hint
                elif key == ord('r'):
                    self.bbox = None
                elif key == 13 or key == 32: 
                    self.process_submission()
                    
            elif self.state == "RESULT":
                if key == ord('n'):
                    self.load_next_level()
                elif key == ord('r'):
                    self.reset_level_state()

        cv2.destroyAllWindows()

if __name__ == "__main__":
    game = MedAIGame()
    game.run()