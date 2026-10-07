"""
Оценка качества RAG системы через RAGAS для assistant_api.
Использует OpenAI API для RAG и для метрик RAGAS.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate

# Правильный импорт для RAGAS 0.4.x - используем классы метрик
try:
    # Новый способ импорта (RAGAS 0.4+)
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.metrics._context_precision import (
        ContextPrecision,
        ContextPrecisionPrompt,
        QAC,
        Verification,
    )
    from ragas.metrics._answer_relevance import AnswerRelevancy
    faithfulness = Faithfulness
    context_precision = ContextPrecision
    answer_relevancy = AnswerRelevancy
except ImportError:
    try:
        # Альтернативный импорт из collections
        from ragas.metrics.collections import faithfulness, context_precision, answer_relevancy
    except ImportError:
        # Fallback на старый импорт
        from ragas.metrics import (
            Faithfulness as faithfulness,
            ContextPrecision as context_precision,
            AnswerRelevancy as answer_relevancy,
        )

from rag_pipeline import RAGPipeline


class PartialContextPrecisionPrompt(ContextPrecisionPrompt):
    """Чанк полезен, если подтверждает любую часть эталона, а не весь ответ целиком."""

    instruction: str = (
        "Given question, answer and context, verify whether the context supports "
        "any part of the given answer. Give verdict as \"1\" if the context supports "
        "at least one claim in the answer, even when the rest of the answer comes "
        "from other contexts. Give verdict as \"0\" only when the context is "
        "irrelevant to the answer. Reply with json output."
    )
    examples = ContextPrecisionPrompt.examples + [
        (
            QAC(
                question="Which drinks does the cafe offer for breakfast and for the evening?",
                context="The breakfast menu includes a plain oat latte.",
                answer=(
                    "Breakfast includes a plain oat latte. "
                    "The evening menu includes a mint tea."
                ),
            ),
            Verification(
                reason=(
                    "The context supports the breakfast part of the answer. "
                    "It does not mention the evening drink, but a context is useful "
                    "when it supports any part of the answer."
                ),
                verdict=1,
            ),
        ),
    ]


# Вопросы и эталонные ответы по loveself-products.txt, написанные независимо от пайплайна.
# Часть вопросов требует двух продуктов сразу. ground_truth не копируется из ответа системы.
EVALUATION_SET = [
    {
        "question": "I travel a lot and hate stirring powder into a drink. I care about my skin, not my joints. Which collagen fits, and why not the bovine one?",
        "ground_truth": (
            "Glow Within is the marine collagen and vitamin capsule for skin, hair and nails. "
            "Marine collagen is the usual choice for skin-focused goals rather than joints or bone, "
            "and the capsule format suits people who dislike mixing powders or travel frequently. "
            "Glow Boost+ is the bovine collagen powder, aimed at connective tissue and bone as well as skin."
        ),
    },
    {
        "question": "If I want a higher collagen dose than a capsule, which marine product is that, and how is it prepared for absorption?",
        "ground_truth": (
            "Glow Within+ is a hydrolysed marine collagen peptide powder with added vitamins, "
            "for women who want a higher dose than the Glow Within capsules provide. "
            "Hydrolysed peptides are broken into smaller fragments for absorption. "
            "It is a flavoured powder meant to be drunk: 315g, 30 servings, about one month."
        ),
    },
    {
        "question": "I want a protein shake I can stir into porridge and fruit smoothies without a strong flavour. Which one is that, and does the range say anything about eating less on Wegovy?",
        "ground_truth": (
            "Protein Power Vanilla is the neutral flavour of the two and blends into fruit smoothies, "
            "porridge and baking. Both Protein Power shakes are taken to maintain protein intake and "
            "protect muscle while eating less on a GLP-1 medication such as Mounjaro or Wegovy. "
            "Chocolate is the richer flavour and suits mixing with milk or water alone."
        ),
    },
    {
        "question": "When someone on a GLP-1 drug is eating less and their fibre intake drops, which products in the range are described as covering protein and fibre?",
        "ground_truth": (
            "Protein Power Chocolate and Protein Power Vanilla are taken to hit protein targets and "
            "protect muscle while eating less on Mounjaro or Wegovy. Tummy Tamer, a fibre and probiotic "
            "powder, is included in the GLP-1 Support pack because fibre intake commonly drops. "
            "Its prebiotic fibre also supports fullness."
        ),
    },
    {
        "question": "I am going through menopause, losing muscle, and I also want help with focus. Is there a product for that beyond a typical gym supplement?",
        "ground_truth": (
            "Mind & Muscle is creatine monohydrate for muscle strength and brain energy. "
            "It is taken to build and preserve lean muscle, reduce age-related muscle loss, and support "
            "mental focus, memory and cognitive energy. The text says creatine is increasingly taken by "
            "women, including through perimenopause and menopause when muscle mass declines. "
            "Each serving is 5g pure creatine monohydrate with no fillers or additives."
        ),
    },
    {
        "question": "I need one product for a racing mind at bedtime and another for feeling wired during the workday. Both involve ashwagandha — when is each one taken, and what else is in them?",
        "ground_truth": (
            "Deep Sleep combines magnesium and ashwagandha and is taken in the evening before bed "
            "for difficulty falling asleep, waking in the night, and a racing mind. Magnesium supports "
            "muscle relaxation and nervous-system calm. Tranquil Mind is a daytime calm complex of "
            "ashwagandha root, botanicals, functional mushrooms and B vitamins, taken in the morning "
            "for ongoing stress and feeling overwhelmed or wired."
        ),
    },
    {
        "question": "I cannot swallow capsules and I suspect I am low on vitamin D. What should I look at, and what is the K2 in it for?",
        "ground_truth": (
            "Sunshine Support+ is a liposomal liquid of vitamin D3 with K2, and the liquid form suits "
            "anyone who cannot swallow capsules. It is taken for vitamin D deficiency, described as "
            "widespread in the UK where sunlight is limited for much of the year. K2 directs calcium "
            "toward bone. One bottle is 60ml, up to two months."
        ),
    },
    {
        "question": "After antibiotics I want a women's probiotic I do not have to mix. How is that different from the powder aimed at bloating and constipation?",
        "ground_truth": (
            "Gut Guardian is a daily probiotic capsule for women, 20 billion CFU across 8 strains, "
            "one capsule a day with no mixing and no taste. It is taken to restore gut flora after "
            "antibiotics and to support vaginal and urinary health. Tummy Tamer is a fibre and probiotic "
            "powder for a bloated stomach, constipation, irregular bowel movements and IBS-type discomfort, "
            "and to increase daily fibre. It is not described as the post-antibiotic capsule."
        ),
    },
]


def prepare_dataset(pipeline: RAGPipeline, items: list) -> Dataset:
    """
    Подготовка датасета для RAGAS из вопросов и эталонных ответов.
    
    Args:
        pipeline: RAG pipeline для получения ответов
        items: список пар question / ground_truth
    
    Returns:
        Dataset для RAGAS с полями: question, answer, contexts, ground_truth
    """
    questions_list = []
    answers_list = []
    contexts_list = []
    ground_truths_list = []
    
    print("[*] Получение ответов от RAG системы...\n")
    
    for i, item in enumerate(items, 1):
        question = item["question"]
        print(f"  {i}/{len(items)}: {question}")
        
        # Получаем ответ от RAG системы (без использования кеша)
        result = pipeline.query(question, use_cache=False)
        
        # Формируем данные для RAGAS
        questions_list.append(question)
        answers_list.append(result["answer"])
        
        # Контекст - список текстов из найденных документов
        context_texts = [doc["text"] for doc in result["context_docs"]]
        contexts_list.append(context_texts)
        
        # Эталонный ответ, заданный заранее, а не взятый из ответа системы
        ground_truths_list.append(item["ground_truth"])
        
        print(f"     [+] Ответ получен от OpenAI API")
    
    print()
    
    # Создаём датасет для RAGAS
    dataset_dict = {
        "question": questions_list,
        "answer": answers_list,
        "contexts": contexts_list,
        "ground_truth": ground_truths_list
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    return dataset


def evaluate_rag_system():
    """
    Основная функция оценки RAG-системы через RAGAS.
    
    Процесс:
    1. Инициализация RAG pipeline
    2. Генерация ответов на тестовые вопросы
    3. Подготовка датасета для RAGAS
    4. Запуск оценки метрик
    5. Вывод результатов
    """
    print("=" * 70)
    print("ОЦЕНКА КАЧЕСТВА RAG-СИСТЕМЫ (API MODE) ЧЕРЕЗ RAGAS")
    print("=" * 70)
    print()
    
    # Проверка наличия API ключа
    if not os.getenv("OPENAI_API_KEY"):
        print("[ОШИБКА] OPENAI_API_KEY не установлен")
        print("\nУстановите переменную окружения:")
        print("  Windows (PowerShell): $env:OPENAI_API_KEY='your-key'")
        print("  Windows (CMD): set OPENAI_API_KEY=your-key")
        print("  Linux/Mac: export OPENAI_API_KEY='your-key'")
        print("\nИли создайте файл .env в корне проекта с содержимым:")
        print("  OPENAI_API_KEY=your-key-here")
        sys.exit(1)
    
    # Инициализация RAG pipeline
    try:
        print("[*] Инициализация RAG системы (API mode)...\n")
        pipeline = RAGPipeline(
            collection_name="loveself_collection",
            cache_db_path="api_rag_cache.db",
            data_file="data/loveself-products.txt",
            model="gpt-4o-mini"
        )
        print("\n[OK] RAG система готова к оценке\n")
    except Exception as e:
        print(f"[ОШИБКА] Ошибка инициализации RAG pipeline: {e}")
        sys.exit(1)
    
    # Подготовка датасета
    print("=" * 70)
    dataset = prepare_dataset(pipeline, EVALUATION_SET)
    print("=" * 70)
    
    print("\n[*] Запуск оценки метрик RAGAS...")
    print("   Метрики: Faithfulness, Context Precision, Answer Relevancy")
    print("   Context Precision: фрагмент полезен, если подтверждает любую часть эталона")
    print("   Судья метрик: gpt-4.1 (temperature=0)")
    print("   Ответы RAG по-прежнему генерирует gpt-4o-mini")
    print("   (это займёт несколько минут: RAGAS вызывает OpenAI для оценки и embeddings)\n")
    
    # ВАЖНО: инициализируем метрики как объекты (с круглыми скобками)
    print("   [+] Используем базовые метрики RAGAS\n")
    metrics_to_use = [
        faithfulness(),
        context_precision(context_precision_prompt=PartialContextPrecisionPrompt()),
        answer_relevancy(),
    ]
    # Детерминированный судья. Генерация ответов остаётся в RAGPipeline.
    judge_llm = ChatOpenAI(model="gpt-4.1", temperature=0)
    
    # Запускаем оценку RAGAS
    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics_to_use,
            llm=judge_llm
        )
    except Exception as e:
        print(f"[ОШИБКА] Ошибка при оценке: {e}")
        sys.exit(1)
    
    # Обработка и вывод результатов
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 70)
    
    # Вычисляем средние значения метрик (игнорируя NaN)
    import math
    
    faithfulness_values = [
        v for v in result['faithfulness'] 
        if not (isinstance(v, float) and math.isnan(v))
    ]
    context_precision_values = [
        v for v in result['context_precision'] 
        if not (isinstance(v, float) and math.isnan(v))
    ]
    answer_relevancy_values = [
        v for v in result['answer_relevancy']
        if not (isinstance(v, float) and math.isnan(v))
    ]
    
    avg_faithfulness = (
        sum(faithfulness_values) / len(faithfulness_values) 
        if faithfulness_values else 0
    )
    avg_context_precision = (
        sum(context_precision_values) / len(context_precision_values) 
        if context_precision_values else 0
    )
    avg_answer_relevancy = (
        sum(answer_relevancy_values) / len(answer_relevancy_values)
        if answer_relevancy_values else 0
    )
    
    # Выводим общие метрики
    print()
    print("[МЕТРИКИ] Средние значения:")
    print(f"   Faithfulness (точность ответа):          {avg_faithfulness:.4f}")
    print(f"   Context Precision (точность контекста):  {avg_context_precision:.4f}")
    print(f"   Answer Relevancy (релевантность ответа): {avg_answer_relevancy:.4f}")
    
    # Вычисляем и выводим средний балл
    avg_score = (avg_faithfulness + avg_context_precision + avg_answer_relevancy) / 3
    print(f"\n{'='*70}")
    print(f"[ИТОГО] Средний балл: {avg_score:.4f}")
    
    # Оценка качества системы
    if avg_score >= 0.7:
        print("   Оценка: Отличное качество! [OK]")
        print("   Система показывает высокую точность и релевантность ответов.")
    elif avg_score >= 0.5:
        print("   Оценка: Удовлетворительное качество [!]")
        print("   Рекомендуется улучшить качество документов или промптов.")
    else:
        print("   Оценка: Требует значительного улучшения [X]")
        print("   Необходимо пересмотреть стратегию chunking или качество данных.")
    
    # Выводим детали по каждому вопросу
    print("\n" + "=" * 70)
    print("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ПО ВОПРОСАМ")
    print("=" * 70)
    
    for i, item in enumerate(EVALUATION_SET):
        print(f"\n{i+1}. {item['question']}")
        
        # Faithfulness
        faith_val = result['faithfulness'][i]
        if not (isinstance(faith_val, float) and math.isnan(faith_val)):
            print(f"   Faithfulness:       {faith_val:.4f}")
        else:
            print(f"   Faithfulness:       не удалось вычислить")
        
        # Context Precision
        cp_val = result['context_precision'][i]
        if not (isinstance(cp_val, float) and math.isnan(cp_val)):
            print(f"   Context Precision:  {cp_val:.4f}")
        else:
            print(f"   Context Precision:  не удалось вычислить")

        # Answer Relevancy
        ar_val = result['answer_relevancy'][i]
        if not (isinstance(ar_val, float) and math.isnan(ar_val)):
            print(f"   Answer Relevancy:   {ar_val:.4f}")
        else:
            print(f"   Answer Relevancy:   не удалось вычислить")
    
    # Пояснения к метрикам
    print("\n" + "=" * 70)
    print("[INFO] ПОЯСНЕНИЯ К МЕТРИКАМ")
    print("=" * 70)
    print("""
Faithfulness (Точность ответа):
  Измеряет, насколько ответ соответствует предоставленному контексту.
  Значения: 0.0 - 1.0 (1.0 = полное соответствие контексту)

Context Precision (Точность контекста):
  Оценивает, стоят ли полезные фрагменты выше бесполезных.
  Фрагмент полезен, если подтверждает любую часть эталонного ответа,
  даже когда остальные части лежат в других фрагментах.
  Значения: 0.0 - 1.0 (1.0 = все полезные фрагменты выше бесполезных)

Answer Relevancy (Релевантность ответа):
  Генерирует вопросы, на которые подходит ответ, и сравнивает их с исходным
  вопросом через embeddings.
  Значения: 0.0 - 1.0 (1.0 = ответ строго по вопросу)
    """)
    
    print("=" * 70)
    print("[OK] Оценка завершена!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    evaluate_rag_system()

