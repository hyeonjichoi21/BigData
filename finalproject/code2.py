import requests
from bs4 import BeautifulSoup
import re
import time

# ✅ URL에서 상품번호 추출
def extract_goods_no_from_url(url):
    match = re.search(r'/goods/(\d+)', url, re.IGNORECASE)
    return int(match.group(1)) if match else None

# ✅ 리뷰 크롤링
def crawl_yes24_reviews(goods_no, max_pages=10):
    reviews = []
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": f"https://www.yes24.com/Product/Goods/{goods_no}",
        "X-Requested-With": "XMLHttpRequest",
    }

    for page in range(1, max_pages + 1):
        url = f"https://www.yes24.com/Product/communityModules/GoodsReviewList/{goods_no}?PageNumber={page}"
        res = requests.get(url, headers=headers)

        if res.status_code != 200:
            print(f"❌ 요청 실패: {url}")
            continue

        soup = BeautifulSoup(res.text, "html.parser")
        review_tags = soup.select(".review_cont > p")

        if not review_tags:
            print(f"⚠️ {page}페이지에 리뷰가 없습니다.")
            continue

        for tag in review_tags:
            text = tag.get_text(strip=True)
            if text:
                reviews.append(text)

        time.sleep(1)

    return reviews

# ✅ 실행부
book_url = input("YES24 책 링크를 입력하세요: ")
goods_no = extract_goods_no_from_url(book_url)

if goods_no:
    print(f"\n[검색된 상품 번호: {goods_no}] 리뷰 수집 중...\n")
    reviews = crawl_yes24_reviews(goods_no, max_pages=5)
    print(f"[YES24] 리뷰 {len(reviews)}개 수집 완료!\n")

    from konlpy.tag import Okt
    import re

    stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
    okt = Okt()

    def preprocess_reviews(reviews):
        processed = []
        for review in reviews:
            review = re.sub(r'[^가-힣\s]', '', review)
            tokens = okt.nouns(review)
            tokens = [word for word in tokens if word not in stopwords and len(word) > 1]
            processed.append(" ".join(tokens))
        return processed

    cleaned_reviews = preprocess_reviews(reviews)

    # ✅ 감성 사전
    senti_dict = {
        "좋다": 2, "좋아": 2, "최고": 2, "만족": 2, "감동": 2, "추천": 2, "강추":4,
        "재미": 2, "꿀잼": 3, "재밌": 2, "명작": 5, "인생책": 6,
        "별로": -3, "실망": -3, "지루": -3, "비추": -3, "비추천":-3, "비추다": -3,
        "최악": -4, "불만": -2, "아쉽": -2, "아깝": -2, "쓰레기": -5,
        "별점": -1, "후회": -3, "지저분": -3, "낭비": -4
    }

    def get_sentiment_score(review):
        return sum(senti_dict.get(word, 0) for word in review.split())

    sentiment_results = [{"text": r, "score": get_sentiment_score(r)} for r in cleaned_reviews]

    pos_cnt = sum(1 for r in sentiment_results if r["score"] > 0)
    neg_cnt = sum(1 for r in sentiment_results if r["score"] < 0)
    total_cnt = pos_cnt + neg_cnt if (pos_cnt + neg_cnt) > 0 else 1

    pos_ratio = pos_cnt / total_cnt
    neg_ratio = neg_cnt / total_cnt

    # ✅ TF-IDF
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(cleaned_reviews)
    tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), X.sum(axis=0).tolist()[0]))

    # ✅ 감정-색상 매핑
    emotion_color_map = {
    # 💗 사랑, 관계, 따뜻함
    "사랑": "pink",
    "연애": "lightpink",
    "용기": "salmon",
    "공감": "lightcoral",
    "행복": "gold",
    "희망": "orange",
    "이해": "lightskyblue",
    "성장": "lightgreen",

    # 💔 상실, 고통, 우울
    "이별": "skyblue",
    "우울증": "purple",
    "자살": "midnightblue",
    "허무": "gray",
    "미움": "darkred",
    "상처": "indianred",
    "고독": "dimgray",
    "불안": "blue",
    "죽음": "black",

    # 🤯 철학적·내면적 사고
    "자신": "teal",
    "자아": "darkcyan",
    "철학자": "slategray",
    "질문": "steelblue",
    "생각": "deepskyblue",
    "기억": "mediumorchid",
    "기록": "mediumslateblue",
    "망각": "darkslateblue",
    "내면": "plum",

    # 🧠 인간·사회 구조
    "사람": "lightgray",
    "우리": "lightsteelblue",
    "사회": "darkolivegreen",
    "현실": "cadetblue",
    "시대": "sienna",
    "가족": "khaki",
    "여성": "orchid",

    # 📚 문학, 문체, 예술
    "소설": "burlywood",
    "작가": "wheat",
    "문장": "bisque",
    "문체": "antiquewhite",
    "이야기": "moccasin",

    # 🎭 감성 및 감각
    "감정": "lightcoral",
    "감성": "hotpink",
    "감각": "thistle",
    "관점": "lightseagreen",
    "시선": "aquamarine",

    # 💡 기타 주제/표현
    "기대": "peachpuff",
    "역할": "rosybrown",
    "모습": "tan",
    "순간": "lightsalmon",
    "변화": "mediumaquamarine",
    "인생": "palegreen",
    "행복": "gold",
    "내용": "azure",
    "의미": "lightblue",
    "기억": "mediumorchid",
    
    # 기타
    "페미니즘" : "purple",
    }

    # ✅ 색상 함수 정의
    def custom_color_func(word, font_size, position, orientation, font_path, random_state):
        return emotion_color_map.get(word, "gray")

    # ✅ WordCloud 시각화
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    wordcloud = WordCloud(
        font_path='malgun.ttf',
        background_color='white',
        width=800,
        height=400,
        color_func=custom_color_func
    ).generate_from_frequencies(tfidf_scores)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("🎨 감정 기반 색상 WordCloud")
    plt.show()

else:
    print("책 링크가 잘못되었거나 상품번호 추출에 실패했습니다.")




# GoodsReviewList
# 왜 나 너 사랑: https://www.yes24.com/product/goods/115275383
# 표백: https://www.yes24.com/product/goods/93375712
# 아몬드: https://www.yes24.com/product/goods/37300128
# 난쏘공: https://www.yes24.com/product/goods/125020220
# 망각일기: https://www.yes24.com/product/goods/115843545
# 미움받을용기: https://www.yes24.com/product/goods/116599423
# 파과: https://www.yes24.com/product/goods/125761518 <- 이거 테스트로 ㄱㅊ
# 아가미: https://www.yes24.com/product/goods/125761510 <- 이건 결과 ㅂㄹ임 