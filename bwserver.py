from flask import Flask, request, jsonify
import sdxl
import requests
from konlpy.tag import Okt
from flask import Flask, request, jsonify #크롤링 라이브러리
from flask_cors import CORS

def translate_to_english(text):
    if not text:
        raise ValueError("The input text cannot be empty.")
    papago_url = "https://openapi.naver.com/v1/papago/n2mt" # api주소추가
    papago_headers = {
        "X-Naver-Client-Id": "U01qEPdMcU55vqfUOMsj", #번역사이트 아이디
        "X-Naver-Client-Secret": "QvIU1rSU19"
    }
    data = {
        "source": 'ko',
        "target": 'en',
        "text": text
    }
    response = requests.post(papago_url, headers=papago_headers, data=data)#번역api

    if response.status_code != 200:
        raise Exception(f"Translation request failed with status code {response.status_code}. Error message: {response.text}")

    response_json = response.json()

    if ("message" not in response_json or 
       "result" not in response_json["message"] or 
       "translatedText" not in response_json["message"]["result"]):
       
       raise Exception("The translation API returned an unexpected result.")

    
    translated_text = response_json["message"]["result"]["translatedText"]

    return translated_text


def extract_nouns(text):
    if not text:
        raise ValueError("The input text cannot be empty.")
    
    okt = Okt()
    nouns = okt.nouns(text)
   
    return nouns


app = Flask(__name__)
CORS(app)

@app.route('/generate_image', methods=['POST'])
def generate_image():
    user_input = request.form.get('user_input')

    client = sdxl.ImageGenerator()
    nouns_ko= extract_nouns(user_input)
    nouns_en= [translate_to_english(noun) for noun in nouns_ko]
    prompt_en=", ".join(nouns_en)

    images = client.gen_image(prompt=prompt_en, count=1, width=1024, height=1024,
                              refine="expert_ensemble_refiner", scheduler="DDIM", guidance_scale=7.5,
                              high_noise_frac=0.8, prompt_strength=0.8, num_inference_steps=50)

    return jsonify(images)  # 이미지 정보를 불러옵니다.

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
