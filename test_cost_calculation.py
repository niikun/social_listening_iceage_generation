#!/usr/bin/env python3
# テスト用スクリプト：更新されたコスト計算のテスト

import sys
import os
sys.path.append(os.path.dirname(__file__))

from app import CostTracker

def test_cost_calculation():
    print("=== GPT-4o-mini 料金計算テスト ===")
    
    # CostTrackerを初期化
    cost_tracker = CostTracker()
    
    print(f"入力トークン単価: ${cost_tracker.gpt4o_mini_input_cost:.6f} / 1000トークン")
    print(f"出力トークン単価: ${cost_tracker.gpt4o_mini_output_cost:.6f} / 1000トークン")
    
    # 典型的な回答生成のコスト計算
    print(f"\n--- 100回答のコスト計算 ---")
    
    # 1回答あたりの想定トークン数
    input_tokens_per_response = 200   # プロンプト + ペルソナ情報
    output_tokens_per_response = 50   # 100文字程度の回答
    
    # 100回答分を計算
    total_input_tokens = input_tokens_per_response * 100
    total_output_tokens = output_tokens_per_response * 100
    
    cost_tracker.add_usage(total_input_tokens, total_output_tokens)
    total_cost_usd = cost_tracker.get_total_cost()
    total_cost_jpy = total_cost_usd * 150  # USD to JPY (150円/ドル想定)
    
    print(f"入力トークン: {total_input_tokens:,} tokens")
    print(f"出力トークン: {total_output_tokens:,} tokens")
    print(f"総コスト: ${total_cost_usd:.6f}")
    print(f"総コスト: {total_cost_jpy:.2f}円")
    print(f"1回答あたり: {total_cost_jpy/100:.3f}円")
    
    # AI分析のコスト計算
    print(f"\n--- AI分析のコスト計算 ---")
    
    analysis_input_tokens = 2000   # 質問 + 100回答分
    analysis_output_tokens = 3000  # 2400文字の詳細分析
    
    analysis_cost_usd = (analysis_input_tokens * cost_tracker.gpt4o_mini_input_cost + 
                        analysis_output_tokens * cost_tracker.gpt4o_mini_output_cost) / 1000
    analysis_cost_jpy = analysis_cost_usd * 150
    
    print(f"入力トークン: {analysis_input_tokens:,} tokens")
    print(f"出力トークン: {analysis_output_tokens:,} tokens")
    print(f"分析コスト: ${analysis_cost_usd:.6f}")
    print(f"分析コスト: {analysis_cost_jpy:.2f}円")
    
    # 検索要約のコスト計算
    print(f"\n--- 検索要約のコスト計算 ---")
    
    summary_input_tokens = 500    # 検索結果 + 質問
    summary_output_tokens = 150   # 300文字の要約
    
    summary_cost_usd = (summary_input_tokens * cost_tracker.gpt4o_mini_input_cost + 
                       summary_output_tokens * cost_tracker.gpt4o_mini_output_cost) / 1000
    summary_cost_jpy = summary_cost_usd * 150
    
    print(f"入力トークン: {summary_input_tokens:,} tokens")
    print(f"出力トークン: {summary_output_tokens:,} tokens")
    print(f"要約コスト: ${summary_cost_usd:.6f}")
    print(f"要約コスト: {summary_cost_jpy:.2f}円")
    
    # 完全な調査のトータルコスト
    print(f"\n--- フル調査のトータルコスト ---")
    full_cost_jpy = total_cost_jpy + analysis_cost_jpy + summary_cost_jpy
    print(f"100回答 + AI分析 + 検索要約: {full_cost_jpy:.2f}円")
    
    print(f"\n✅ コスト計算テスト完了")

if __name__ == "__main__":
    test_cost_calculation()