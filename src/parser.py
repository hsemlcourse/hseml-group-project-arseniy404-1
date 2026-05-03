import cloudscraper
import pandas as pd
from bs4 import BeautifulSoup
import time
from tqdm import tqdm

# Создаем сессию cloudscraper один раз
scraper = cloudscraper.create_scraper(
    browser={
        'browser': 'chrome',
        'platform': 'windows',
        'mobile': False
    }
)


def get_problems():
    """Получает список задач через API с обработкой ошибок."""
    url = "https://codeforces.com/api/problemset.problems"

    try:
        # API обычно не требует обхода Cloudflare
        response = scraper.get(url, timeout=15)
        if response.status_code != 200:
            print(f"Ошибка API: статус {response.status_code}")
            return None

        data = response.json()
        problems = data['result']['problems']
        stats = data['result']['problemStatistics']

        stats_dict = {(s['contestId'], s['index']): s['solvedCount'] for s in stats}

        rows = []
        for p in problems:
            if 'rating' not in p: 
                continue
            key = (p['contestId'], p['index'])
            rows.append({
                'contestId': p['contestId'],
                'index': p['index'],
                'name': p['name'],
                'rating': p['rating'],
                'tags': p['tags'],
                'solvedCount': stats_dict.get(key, 0)
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"Критическая ошибка при получении списка: {e}")
        return None


def get_problem_text(contestId, index):
    """Получает текст задачи, обходя Cloudflare защиту."""
    url = f"https://codeforces.com/problemset/problem/{contestId}/{index}"

    try:
        # Используем cloudscraper вместо requests
        response = scraper.get(url, timeout=15)

        if response.status_code != 200:
            print(f"Ошибка {response.status_code}: {url}")
            return None

        soup = BeautifulSoup(response.text, 'html.parser')

        # Ищем условие задачи
        statement = soup.find('div', class_='problem-statement')
        if not statement:
            statement = soup.find('div', class_='ttypography')

        if not statement:
            print(f"Не найден текст задачи: {url}")
            return None

        # Удаляем лишние элементы
        for unwanted in statement.find_all(['div', 'div'],
                                           class_=['header', 'input-specification',
                                                   'output-specification', 'sample-tests',
                                                   'note', 'document']):
            unwanted.decompose()

        # Получаем текст
        text = statement.get_text(separator='\n', strip=True)

        # Очищаем от лишних пробелов
        text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())

        return text if len(text) > 50 else None

    except Exception as e:
        print(f"Ошибка при парсинге {url}: {e}")
        return None


# Основной код
print("Шаг 1: Получаем список всех задач...")
df = get_problems()

if df is not None:
    # Для теста возьмем меньше задач
    sample_df = df.head(1500).copy()  # Начните с 50 для теста
    print(f"Шаг 2: Парсим тексты для {len(sample_df)} задач...")

    texts = []
    success_count = 0

    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        text = get_problem_text(row['contestId'], row['index'])
        texts.append(text)
        if text is not None:
            success_count += 1
        time.sleep(1.0)  # Важно: задержка чтобы не заблокировали

    sample_df['statement'] = texts

    print(f"\nУспешно получено: {success_count}/{len(sample_df)} задач")

    # Сохраняем результат
    sample_df.to_csv('../data/raw/full_problems.csv', index=False)

    # Показываем пример успешного парсинга
    first_success = sample_df[sample_df['statement'].notna()].head(1)
    if not first_success.empty:
        print("\nПример успешно распарсенной задачи:")
        print(f"Название: {first_success['name'].values[0]}")
        print(f"Рейтинг: {first_success['rating'].values[0]}")
        print(f"Текст (первые 200 символов): {first_success['statement'].values[0][:200]}...")
else:
    print("Не удалось получить данные с Codeforces.")
