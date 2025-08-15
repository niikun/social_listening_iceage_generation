#!/usr/bin/env python3
# テスト用スクリプト：就職氷河期世代特化の要約機能テスト

import sys
import os
import asyncio
sys.path.append(os.path.dirname(__file__))

from app import SimulationProvider

async def test_ice_age_summary():
    print("=== 就職氷河期世代特化検索要約テスト ===")
    
    provider = SimulationProvider()
    
    # 雇用・就職関連の質問
    employment_questions = [
        "就職支援の充実について",
        "非正規雇用の問題について", 
        "働き方改革について",
        "転職活動の支援について"
    ]
    
    # 経済・家計関連の質問
    economic_questions = [
        "家計負担軽減について",
        "税制改革について",
        "年金制度について",
        "経済格差について"
    ]
    
    # 社会保障・制度関連の質問
    social_questions = [
        "社会保障制度について",
        "政府の政策について",
        "医療保険制度について"
    ]
    
    # 教育・次世代関連の質問
    education_questions = [
        "子どもの教育について",
        "若者支援について",
        "次世代への投資について"
    ]
    
    # デジタル・技術関連の質問
    tech_questions = [
        "デジタル化について",
        "AI技術について",
        "DX推進について"
    ]
    
    # 一般的な質問
    general_questions = [
        "地域活性化について",
        "環境問題について",
        "国際情勢について"
    ]
    
    all_categories = [
        ("雇用・就職関連", employment_questions),
        ("経済・家計関連", economic_questions),
        ("社会保障・制度関連", social_questions),
        ("教育・次世代関連", education_questions),
        ("デジタル・技術関連", tech_questions),
        ("一般的な質問", general_questions)
    ]
    
    # 各カテゴリーの要約をテスト
    for category_name, questions in all_categories:
        print(f"\n{'='*60}")
        print(f"{category_name}")
        print(f"{'='*60}")
        
        for question in questions:
            print(f"\n【質問】{question}")
            print("-" * 40)
            
            # ダミーの検索結果
            dummy_search_results = [
                {"title": f"{question}に関する記事1", "snippet": "専門家の見解..."},
                {"title": f"{question}に関する記事2", "snippet": "政府の方針..."}
            ]
            
            # 要約実行
            result = await provider.summarize_search_results(dummy_search_results, question)
            
            if result['success']:
                print(result['summary'])
            else:
                print(f"エラー: {result.get('summary', '不明')}")
    
    print("\n✅ 就職氷河期世代特化検索要約テスト完了")

if __name__ == "__main__":
    asyncio.run(test_ice_age_summary())