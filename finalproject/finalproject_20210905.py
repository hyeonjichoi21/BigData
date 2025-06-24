import requests
from bs4 import BeautifulSoup
import re
import time

# ✅ URL에서 상품번호 추출
def extract_goods_no_from_url(url):
    match = re.search(r'/goods/(\d+)', url, re.IGNORECASE)
    if match:
        return int(match.group(1))
    else:
        print("❌ URL에서 상품번호를 추출할 수 없습니다.")
        return None

# ✅ 리뷰 본문 크롤링
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

# ✅ 평점(별점) 크롤링
def crawl_yes24_ratings(goods_no, max_pages=5):
    ratings = []
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
        rating_spans = soup.select("span.total_rating")

        if not rating_spans:
            print(f"⚠️ {page}페이지에 평점 정보가 없습니다.")
            continue

        for span in rating_spans:
            text = span.get_text(strip=True)  # 예: "평점2점"
            match = re.search(r"평점(\d)점", text)
            if match:
                ratings.append(int(match.group(1)))

        time.sleep(1)

    return ratings

# ✅ 컬러맵 선택 함수
def get_colormap_by_rating(ratings):
    if not ratings:
        return "gray"
    avg = sum(ratings) / len(ratings)
    print(f"\n📊 평균 별점: {round(avg, 2)}점")
    if avg >= 8:
        return "Reds"
    elif avg >= 6:
        return "Oranges"
    else:
        return "Blues"




# ✅ 실행부
book_url = input("YES24 책 링크를 입력하세요: ")
goods_no = extract_goods_no_from_url(book_url)

if goods_no:
    print(f"\n[검색된 상품 번호: {goods_no}] 리뷰 수집 중...\n")
    reviews = crawl_yes24_reviews(goods_no, max_pages=5)
    print(f"[YES24] 리뷰 {len(reviews)}개 수집 완료!\n")
    for i, review in enumerate(reviews[:5]):
        print(f"{i+1}. {review}")

    # ✅ 전처리
    from konlpy.tag import Okt
    import re

    stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
    okt = Okt()

    def preprocess_reviews(reviews):
        processed = []
        for review in reviews:
            review = re.sub(r'[^가-힣\s]', '', review)
            tokens = okt.nouns(review)  # 명사만 추출
            tokens = [word for word in tokens if word not in stopwords and len(word) > 1]
            processed.append(" ".join(tokens))
        return processed

    cleaned_reviews = preprocess_reviews(reviews)
    print(f"\n[전처리된 리뷰 예시]\n")
    for i, review in enumerate(cleaned_reviews[:5]):
        print(f"{i+1}. {review}")

    # 감성 분석
    senti_dict = {
        "좋다": 2, "좋아": 2, "최고": 2, "만족": 2, "감동": 2, "추천": 2, "강추":4,
        "재미": 2, "꿀잼": 3, "재밌": 2, "명작": 5, "인생책": 6,
        "별로": -3, "실망": -3, "지루": -3, "비추": -3, "비추천":-3, "비추다": -3,
        "최악": -4, "불만": -2, "아쉽": -2, "아깝": -2, "쓰레기": -5,
        "별점": -1, "후회": -3, "지저분": -3, "낭비": -4
    }

    def get_sentiment_score(review):
        score = 0
        for word in review.split():
            score += senti_dict.get(word, 0)
        return score

    sentiment_results = [{"text": r, "score": get_sentiment_score(r)} for r in cleaned_reviews]

    print("\n[감성 분석 결과 예시]")
    for i, item in enumerate(sentiment_results[:5]):
        label = "긍정" if item["score"] > 0 else "부정" if item["score"] < 0 else "중립"
        print(f"{i+1}. ({label}) {item['text']}")

    # ✅ 긍/부정 비율 계산
    pos_cnt = sum(1 for r in sentiment_results if r["score"] > 0)
    neg_cnt = sum(1 for r in sentiment_results if r["score"] < 0)
    total_cnt = pos_cnt + neg_cnt if (pos_cnt + neg_cnt) > 0 else 1

    pos_ratio = pos_cnt / total_cnt
    neg_ratio = neg_cnt / total_cnt

    # ✅ 2. TF-IDF 단어 추출
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(cleaned_reviews)

    tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), X.sum(axis=0).tolist()[0]))

    top_keywords = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:20]
    print("\n[TF-IDF 상위 키워드]")
    for word, score in top_keywords:
        print(f"{word}: {round(score, 3)}")

    # ✅ matplotlib 한글 폰트 설정
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    font_path = 'C:/Windows/Fonts/malgun.ttf'
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)

    # ✅ 3. WordCloud 시각화 (감성 기반 컬러)
    from wordcloud import WordCloud

    colormap = "pink" if pos_ratio > neg_ratio else "Blues"

    wordcloud = WordCloud(
        font_path=font_path,
        background_color='white',
        width=800,
        height=400,
        colormap=colormap
    ).generate_from_frequencies(tfidf_scores)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f" 감성 기반 WordCloud ({'긍정 우세' if pos_ratio > neg_ratio else '부정 우세'})")
    plt.show()

    # 감정-색상 매핑 기반 WordCloud
    emotion_color_map = {
        "사랑": "pink", "연애": "lightpink", "용기": "salmon", "공감": "lightcoral", "행복": "gold",
        "희망": "orange", "이해": "lightskyblue", "성장": "lightgreen", "이별": "skyblue", "우울증": "purple",
        "자살": "midnightblue", "허무": "gray", "미움": "darkred", "상처": "indianred", "고독": "dimgray",
        "불안": "blue", "죽음": "black", "자신": "teal", "자아": "darkcyan", "철학자": "slategray",
        "질문": "steelblue", "생각": "deepskyblue", "기억": "mediumorchid", "기록": "mediumslateblue",
        "망각": "darkslateblue", "내면": "plum", "사람": "lightgray", "우리": "lightsteelblue",
        "사회": "darkolivegreen", "현실": "cadetblue", "시대": "sienna", "가족": "khaki",
        "여성": "orchid", "소설": "burlywood", "작가": "wheat", "문장": "bisque", "문체": "antiquewhite",
        "이야기": "moccasin", "감정": "lightcoral", "감성": "hotpink", "감각": "thistle",
        "관점": "lightseagreen", "시선": "aquamarine", "기대": "peachpuff", "역할": "rosybrown",
        "모습": "tan", "순간": "lightsalmon", "변화": "mediumaquamarine", "인생": "palegreen",
        "내용": "azure", "의미": "lightblue", "페미니즘": "purple"
    }

    def custom_color_func(word, font_size, position, orientation, font_path, random_state):
        return emotion_color_map.get(word, "gray")

    emotion_wordcloud = WordCloud(
        font_path=font_path,
        background_color='white',
        width=800,
        height=400,
        color_func=custom_color_func
    ).generate_from_frequencies(tfidf_scores)

    plt.figure(figsize=(10, 5))
    plt.imshow(emotion_wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(" 감정 기반 색상 WordCloud")
    plt.show()
    
    
    # ✅ 실행 후 리뷰 수집이 완료된 이후, 아래 코드 추가

    # ⭐️ 별점 기반 WordCloud 추가
    ratings = crawl_yes24_ratings(goods_no, max_pages=5)
    print(f"\n[YES24] 평점 {len(ratings)}개 수집 완료!")
    print("▶ 평점 예시:", ratings[:10])

    colormap = get_colormap_by_rating(ratings)

    wordcloud_rating_based = WordCloud(
        font_path=font_path,
        background_color='white',
        width=800,
        height=400,
        colormap=colormap
    ).generate_from_frequencies(tfidf_scores)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_rating_based, interpolation='bilinear')
    plt.axis('off')
    plt.title(f" 별점 기반 WordCloud (컬러: {colormap})")
    plt.show()

else:
    print("책 링크가 잘못되었거나 상품번호 추출에 실패했습니다.")




# GoodsReviewList
# 왜 나 너 사랑: https://www.yes24.com/product/goods/115275383  O 
# 표백: https://www.yes24.com/product/goods/93375712
# 아몬드: https://www.yes24.com/product/goods/37300128
# 난쏘공: https://www.yes24.com/product/goods/125020220
# 망각일기: https://www.yes24.com/product/goods/115843545
# 미움받을용기: https://www.yes24.com/product/goods/116599423 O 
# 파과: https://www.yes24.com/product/goods/125761518 <- 얘로 테스트하기
# 아가미: https://www.yes24.com/product/goods/125761510 <- 이건 결과 ㅂㄹ임 
# 모순: https://www.yes24.com/product/goods/8759796
# 나 소망 내게 금지: https://www.yes24.com/product/goods/72127217



