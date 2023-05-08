import requests
import re
import os 
import numpy as np
import cairosvg 
import cv2
from bs4 import BeautifulSoup
from io import BytesIO
from PIL import Image
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager


def scrapSignalKeyword():
    '''
    현재 네이버 시그널의 실시간 검색어를 반환해주는 함수
    '''
    url = 'https://api.signal.bz/news/realtime'
    resp = requests.get(url=url)
    top10dict = resp.json()['top10']
    return [t['keyword'] for t in top10dict]


def cleanArticle(content):
    '''
    네이버 기사의 크롤링 이후 괄호 또는 이미지태그, 이메일 주소등을 제거하는 함수
    
    content : 스크래핑한 기사의 원문
    '''
    cleanr_image = re.compile('<em class="img_desc">.*?</em>')
    cleanr_tag = re.compile('<.*?>')
    cleanr_email = re.compile('([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)')        
    rmve_bracket = re.compile("\(.*\)")
    rmve_bracket2 = re.compile("\[.*\]" )  
    cleantext = re.sub(cleanr_image, '', content)
    cleantext = re.sub(cleanr_tag, '', cleantext)
    cleantext = re.sub(cleanr_email, '', cleantext)
    cleantext = re.sub(rmve_bracket, '', cleantext)
    cleantext = re.sub(rmve_bracket2, '', cleantext)

    return cleantext.strip()

def clean_text(inputString):
    '''
    기사를 txt형식으로 저장시에 저장파일이릉의 특수문자를 제거하기위한 함수

    inputString : str / 특수문자를 제거할 문자열
    '''
    text_rmv = re.sub('[-=+,#/\?:^.@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·]', ' ', inputString)
    return text_rmv.strip()

def scrapNaverNewsKeyword(key, article_num, headers, sort='1', length=350):
    '''
    네이버 뉴스를 검색어를 기준으로 스크래핑하여 (제목, 내용)들의 리스트로 반환하는 함수

    key : str / 검색어
    article_num : int / 스크래핑할 기사의 수
    headers : dict / 유저 에이전트 정보가 담겨있는 딕셔너리
    sort : str / '0' or '1' / 
           '0' -> 기사를 관련도순으로 정렬 후 스크랩, '1' -> 기사를 최신순으로 정렬 후 스크랩
    length : int / 기사를 스크랩시에 클린텍스트 이후 길이가 해당숫자를 넘을 시에만 스크래핑
    '''

    articles = []
    links = [] 
    page_num = 0
    while True:
        url = 'https://search.naver.com/search.naver?where=news'
        page = f'{page_num}1'
        resp = requests.get(url, params={'sm':'tab_jum', 'query':key, 'sort':sort, 'start':page}) 
        news_search = BeautifulSoup(resp.text, 'html.parser') 
        url_list = [d['href'] for d in news_search.find_all('a', attrs={'class':'info'}) if d.text=='네이버뉴스']
        for url in url_list:    
            news = requests.get(url,headers=headers) 
            news_html = BeautifulSoup(news.text,"html.parser") 
            # 뉴스의 종류에 따라 달라지는 페이지 형식을 적용하기위함
            newstype = news.url.split('.')[0].split('//')[-1]

            if newstype == 'sports':
                title = news_html.find('h4', {'class':'title'}).text
                content = news_html.find_all('div', {'id':'newsEndContents'})  ##스포츠 일때는 sid가 없어서 1순위 확인
                content = str(content)
                content = content.split('<p class="source">') 
                text = content[0]            
            elif newstype == 'entertain':
                title = news_html.find('h2', {'class':'end_tit'}).text
                text = ' '.join([paragraph.text for paragraph in news_html.find_all('div', {'id':'articeBody'})]) 
            else:
                title = news_html.find('h2', {'class':'media_end_head_headline'}).text
                text = ' '.join([paragraph.text for paragraph in news_html.find_all('div', {'class':'go_trans _article_content'})]) 

            cleaned_text = cleanArticle(text)
            if len(cleaned_text) > length:
                articles.append((title.strip(), cleaned_text))
                links.append(url)
            if len(articles) == article_num:
                break
        if len(articles) == article_num:
            break
        page_num+=1  
    return articles, links

def saveArticles(articles, path):
    '''
    기사 내용을 담은 기사 제목.txt 을 path에 저장하는 함수

    articles : list / (제목, 기사내용)으로 이루어진 리스트
    path : str / txt파일이 저장될 위치
    '''
    for title, content in articles:
        f = open(os.path.join(path, clean_text(title) + '.txt'), 'w', encoding='utf8')
        f.write(content)
        f.close()

def set_chrome_driver(headers):
    '''
    크롤링을 위한 셀레니움 브라우저를 생성하는 함수

    headers : dict / 유저 에이전트 정보가 담겨있는 딕셔너리
    '''
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument("user-agent="+headers["user-agent"])
    driver = webdriver.Chrome('chromedriver', options=chrome_options)
    return driver

def check_txt_logo(txt, in_=True):
    '''
    문자열에 로고에 해당하는 키워드가 있는지 검사하는 코드

    txt : str / 키워드 체크를 진행할 문자열
    in_ : bool / True -> 키워드가 한개라도 있는지 체크 False -> 키워드가 하나도 없는지 체크
    '''
    # 소문자를 모두 대문자로 통일
    txt = txt.upper()
    if in_:
        return '로고' in  txt or 'LOGO' in  txt or 'CI' in  txt or '휘장' in txt
    else:
        return '로고' not in txt and 'LOGO' not in  txt and 'CI' not in  txt and '휘장' not in txt


def scrapNamuImg(key, path, headers,namuKeyword_kind='person'):
    '''
    원하는 키워드를 활용하여 나무위키에서 원하는 이미지를 스크랩하는 함수

    key : str / 나무위키에 검색할 키워드
    path : str / 스크랩한 이미지가 저장될 위치
    headers : dict / 유저 에이전트 정보가 담겨있는 딕셔너리
    namuKeyword_kind : str / 검색하는 key의 종류 
    '''

    driver = set_chrome_driver(headers)
    url = 'https://namu.wiki/w/'+key
    driver.get(url=url)
    html = driver.page_source
    driver.close()
    soup = BeautifulSoup(html, 'lxml')

    if namuKeyword_kind == 'person':
        # 이미지 이름에 로고에 해당하는 키워드가 들어가 있지 않는 이미지링크들
        imglink = [il['src'] for il in soup.find_all('img', attrs={'class':'XBliG7hv'}) if not il.find_parent('dd') and check_txt_logo(il['alt'], in_=False) ]
        for i, link in enumerate(imglink):
            # 각 이미지 링크에 접근
            res=requests.get("https:"+link,headers=headers)
            # 웹에서 사용하는 svg나 video 형식은 가지고 오지 않도록 
            if 'svg' not in res.text and 'video' not in res.text:
                urlopen_img = Image.open(BytesIO(res.content))
                # counts = np.unique(np.array(urlopen_img.split()[-1]), return_counts=True)[1]
                # ratio = counts[0]/np.sum(counts)
                # 원본이미지의 화소가 기준치를 넘을 시에만 
                if urlopen_img.size[1]*urlopen_img.size[0] > 100000 :
                    print(res.url)
                    if path:
                        urlopen_img.save(path,'png')
                    return urlopen_img
                    break
    else :
        # 보정된 검색어를 가지고오고 소문자를 대문자로 통일
        key_ = soup.find('title').text.replace(' - 나무위키','').upper()
        # 키워드와 로고키워드가 있을 시에
        imglink = [il['src'] for il in soup.find_all('img', attrs={'class':'XBliG7hv'}) if not il.find_parent('dd') and (key_ in  il['alt']) and check_txt_logo(il['alt']) ]
        if len(imglink) < 1:
            #로고키워드가 있을 시에
            imglink = [il['src'] for il in soup.find_all('img', attrs={'class':'XBliG7hv'}) if not il.find_parent('dd') and check_txt_logo(il['alt'])]
        for i, link in enumerate(imglink):
            res=requests.get("https:"+link,headers=headers)
            try:
                # svg형식의 경우 처리를 하여 가지고옴
                urlopen_img = Image.open(BytesIO(cairosvg.svg2png(res.content)))
                print(res.url)
                if path:
                    urlopen_img.save(path,'png')
                return urlopen_img
                break
            except:
                urlopen_img = Image.open(BytesIO(res.content))
                print(res.url)
                if path:
                    urlopen_img.save(path,'png')
                return urlopen_img
                break

def save_article_img(url, headers, save_path, height=300, width=500):
    '''
    네이버 뉴스기사에 있는 이미지를 스크래핑하고 좌우로 원하는 사이즈에 맞게 높이를 맞추고 설정한 너비만큼은 검정색으로 패딩하는 함수
    
    url : str / 스크랩할 이미지가 있는 뉴스의 주소  
    headers : dict / 유저 에이전트 정보가 담겨있는 딕셔너리
    save_path : str / 이미지가 저장될 위치
    height : int / 리턴될 이미지의 높이
    width : int / 리턴될 이미지의 너비
    
    '''  
    try:
        res = requests.get(url,headers=headers) 
        soup = BeautifulSoup(res.text,"html.parser") 
        # 뉴스의 종류에 따라 달라지는 페이지 형식을 적용하기위함
        newstype = res.url.split('.')[0].split('//')[-1]
        if newstype == 'sports' :
            a = soup.find('span', attrs = {'class':'end_photo_org'})
            if a:
                img_url = a.img['src']

        else:
            a = soup.find('img', attrs = {'id':'img1'})
            if a: 
                try:
                    img_url = a['data-src']
                except:
                    img_url = a['src']
        img_res = requests.get(img_url,headers=headers)
        tmp_img = Image.open(BytesIO(img_res.content))
        tmp_img = np.array(tmp_img)
        aspect_ratio = float(height) / tmp_img.shape[0]
        dsize = (int(tmp_img.shape[1] * aspect_ratio), height)

        resized = cv2.resize(tmp_img, dsize, interpolation=cv2.INTER_AREA)

        y,x,h,w = (0,0,resized.shape[0], resized.shape[1])
        if resized.shape[1] > width:
            mid_x = w//2
            offset_x = 250
            img_re = resized[0:resized.shape[0], mid_x-offset_x:mid_x+offset_x]
        else:
            # 그림 주변에 검은색으로 칠하기
            w_x = (width-(w-x))/2  # w_x = (300 - 그림)을 뺀 나머지 영역 크기 [ 그림나머지/2 [그림] 그림나머지/2 ]
            h_y = (height-(h-y))/2

            if(w_x < 0):         # 크기가 -면 0으로 지정.
                w_x = 0
            elif(h_y < 0):
                h_y = 0

            # M = np.float32([[1,0,w_x], [0,1,h_y]])  #(2*3 이차원 행렬)
            # img_re = cv2.warpAffine(resized, M, (width, height))
            pad = np.ones((height,int(w_x),3))*232
            img_re = np.hstack([pad, resized, pad])
        img_re = img_re.astype(np.float32)
        img_re = cv2.cvtColor(img_re, cv2.COLOR_BGR2RGB)
        print(img_url)
        cv2.imwrite(save_path,img_re)
        return img_re

    except:
        bg = np.ones((height, width, 3)) * 232 
        cv2.imwrite(save_path,bg)
        return bg



