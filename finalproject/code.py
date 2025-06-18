import requests
from bs4 import BeautifulSoup
import re
import time

# ✅ URL에서 상품번호 추출 (대소문자 구분 없음)
def extract_goods_no_from_url(url):
    match = re.search(r'/goods/(\d+)', url, re.IGNORECASE)
    if match:
        return int(match.group(1))
    else:
        print("❌ URL에서 상품번호를 추출할 수 없습니다.")
        return None

# ✅ 리뷰 크롤링
def crawl_yes24_reviews(goods_no, max_pages=10):
    reviews = []
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": f"https://www.yes24.com/Product/Goods/{goods_no}",
        "X-Requested-With": "XMLHttpRequest",  # AJAX 요청처럼 보이게
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
            continue  # break 대신 continue로 변경

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
    for i, review in enumerate(reviews[:5]):
        print(f"{i+1}. {review}")

    # ✅ 전처리 코드 여기서 시작
    from konlpy.tag import Okt
    import re

    stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
    okt = Okt()

    def preprocess_reviews(reviews):
        processed = []
        for review in reviews:
            review = re.sub(r'[^가-힣\s]', '', review)
            tokens = okt.morphs(review, stem=True)
            tokens = [word for word in tokens if word not in stopwords and len(word) > 1]
            processed.append(" ".join(tokens))
        return processed

    cleaned_reviews = preprocess_reviews(reviews)
    print(f"\n[전처리된 리뷰 예시]\n")
    for i, review in enumerate(cleaned_reviews[:5]):
        print(f"{i+1}. {review}")
        
    # ✅ 감성 분석 (간단 사전 기반)
    senti_dict = {
    "좋다": 2, "좋아": 2, "최고": 2, "만족": 2, "감동": 2, "추천": 2, "재미": 2, "재밌": 2,
    "별로": -3, "실망": -3, "지루": -3, "비추": -3, "비추천":-3, "비추다": -3, "최악": -4, "불만": -2,
    "아쉽": -2, "아깝": -2, "쓰레기": -5, "별점": -1, "후회": -3, "지저분": -3
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

else:
    print("책 링크가 잘못되었거나 상품번호 추출에 실패했습니다.")


# GoodsReviewList
# 왜 나 너 사랑: https://www.yes24.com/product/goods/115275383
# 표백: https://www.yes24.com/product/goods/93375712
# 아몬드: https://www.yes24.com/product/goods/37300128
# 난쏘공: https://www.yes24.com/product/goods/125020220
# 망각일기: https://www.yes24.com/product/goods/115843545


































