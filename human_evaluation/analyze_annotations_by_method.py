import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from pathlib import Path


def extract_fluency_score(fluency_str):
    if pd.isna(fluency_str):
        return None
    fluency_str = str(fluency_str).strip()
    if '(' in fluency_str:
        num_part = fluency_str.split('(')[0].strip()
        try:
            return int(num_part)
        except:
            return None
    try:
        return int(fluency_str)
    except:
        return None


def extract_imperceptability_score(impercept_str):
    if pd.isna(impercept_str):
        return None
    impercept_str = str(impercept_str).strip()
    if impercept_str.lower() == 'normal':
        return 1
    elif impercept_str.lower() == 'attacked':
        return 0
    else:
        return None


def main():
    annotations_file = Path(__file__).parent / 'annotations' / 'human_annotations.csv'
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    output_file = results_dir / 'annotation_analysis_by_method.csv'
    
    print("=" * 80)
    print("Human Annotation Analysis by Method")
    print("=" * 80)
    
    print(f"\nReading annotations from: {annotations_file}")
    
    with open(annotations_file, 'r', encoding='utf-8') as f:
        header_row1 = f.readline().strip().split(',')
        header_row2 = f.readline().strip().split(',')
    
    df = pd.read_csv(annotations_file, skiprows=[0, 1], header=None)
    
    col_names = ['id', 'doc_id', 'method', 'document']
    
    annotator_start_idx = None
    for i, col in enumerate(header_row1):
        if 'annotator' in col.lower():
            annotator_start_idx = i
            break
    
    if annotator_start_idx is None:
        annotator_start_idx = 4
    
    num_annotators = (len(header_row1) - annotator_start_idx) // 2
    
    annotator_fluency_cols = []
    annotator_impercept_cols = []
    
    for i in range(num_annotators):
        annotator_id = f"annotator_{i+1:02d}"
        fluency_col_name = f'{annotator_id}_fluency'
        impercept_col_name = f'{annotator_id}_imperceptibility'
        
        col_names.append(fluency_col_name)
        col_names.append(impercept_col_name)
        
        annotator_fluency_cols.append(fluency_col_name)
        annotator_impercept_cols.append(impercept_col_name)
    
    df.columns = col_names[:len(df.columns)]
    
    print(f"Found {num_annotators} annotators")
    print(f"Fluency columns: {annotator_fluency_cols}")
    print(f"Imperceptibility columns: {annotator_impercept_cols}")
    
    for i in range(num_annotators):
        annotator_id = f"annotator_{i+1:02d}"
        fluency_idx = 4 + 2 * i
        impercept_idx = 4 + 2 * i + 1
        
        if fluency_idx < len(df.columns) and impercept_idx < len(df.columns):
            df[f'{annotator_id}_fluency'] = df.iloc[:, fluency_idx].apply(extract_fluency_score)
            df[f'{annotator_id}_impercept'] = df.iloc[:, impercept_idx].apply(extract_imperceptability_score)
    
    if 'method' not in df.columns:
        raise ValueError("'method' column not found in data")
    
    df = df[df['method'].notna()].copy()
    
    print(f"\nTotal documents: {len(df)}")
    print(f"Methods found: {sorted(df['method'].unique())}")
    
    results = []
    
    for method in sorted(df['method'].unique()):
        method_data = df[df['method'] == method].copy()
        
        if len(method_data) == 0:
            continue
        
        all_fluency_scores = []
        all_impercept_scores = []
        annotator_pairs_fluency = []
        annotator_pairs_impercept = []
        
        for i in range(len(annotator_fluency_cols)):
            annotator_id = f"annotator_{i+1:02d}"
            fluency_col = f'{annotator_id}_fluency'
            impercept_col = f'{annotator_id}_impercept'
            
            fluency_scores = method_data[fluency_col].dropna()
            impercept_scores = method_data[impercept_col].dropna()
            
            all_fluency_scores.extend(fluency_scores.tolist())
            all_impercept_scores.extend(impercept_scores.tolist())
            
            annotator_pairs_fluency.append(fluency_scores)
            annotator_pairs_impercept.append(impercept_scores)
        
        avg_imperceptibility = np.mean(all_impercept_scores) if all_impercept_scores else np.nan
        avg_fluency = np.mean(all_fluency_scores) if all_fluency_scores else np.nan
        
        kappa_impercept_list = []
        kappa_fluency_list = []
        
        for i in range(len(annotator_pairs_fluency)):
            for j in range(i + 1, len(annotator_pairs_fluency)):
                common_indices = method_data.index.intersection(
                    annotator_pairs_impercept[i].index.intersection(
                        annotator_pairs_impercept[j].index
                    )
                )
                
                if len(common_indices) > 0:
                    impercept_i = annotator_pairs_impercept[i].loc[common_indices]
                    impercept_j = annotator_pairs_impercept[j].loc[common_indices]
                    
                    kappa_imp = cohen_kappa_score(impercept_i.values, impercept_j.values)
                    kappa_impercept_list.append(kappa_imp)
                
                common_indices_fluency = method_data.index.intersection(
                    annotator_pairs_fluency[i].index.intersection(
                        annotator_pairs_fluency[j].index
                    )
                )
                
                if len(common_indices_fluency) > 0:
                    fluency_i = annotator_pairs_fluency[i].loc[common_indices_fluency]
                    fluency_j = annotator_pairs_fluency[j].loc[common_indices_fluency]
                    
                    kappa_fl = cohen_kappa_score(fluency_i.values, fluency_j.values, weights='quadratic')
                    kappa_fluency_list.append(kappa_fl)
        
        avg_kappa_imperceptibility = np.mean(kappa_impercept_list) if kappa_impercept_list else np.nan
        avg_kappa_fluency = np.mean(kappa_fluency_list) if kappa_fluency_list else np.nan
        
        num_documents = len(method_data)
        
        results.append({
            'method': method,
            'num_documents': num_documents,
            'avg_imperceptibility': round(avg_imperceptibility, 2) if not np.isnan(avg_imperceptibility) else np.nan,
            'kappa_imperceptibility': round(avg_kappa_imperceptibility, 2) if not np.isnan(avg_kappa_imperceptibility) else np.nan,
            'avg_fluency': round(avg_fluency, 2) if not np.isnan(avg_fluency) else np.nan,
            'kappa_fluency_weighted': round(avg_kappa_fluency, 2) if not np.isnan(avg_kappa_fluency) else np.nan
        })
        
        print(f"\n{method}:")
        print(f"  Documents: {num_documents}")
        print(f"  Avg Imperceptibility: {avg_imperceptibility:.2f}")
        print(f"  Kappa (Imperceptibility): {avg_kappa_imperceptibility:.2f}")
        print(f"  Avg Fluency: {avg_fluency:.2f}")
        print(f"  Kappa (Fluency, Weighted): {avg_kappa_fluency:.2f}")
    
    results_df = pd.DataFrame(results)
    
    results_df = results_df.sort_values('method')
    
    numeric_cols = ['avg_imperceptibility', 'kappa_imperceptibility', 'avg_fluency', 'kappa_fluency_weighted']
    for col in numeric_cols:
        results_df[col] = results_df[col].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "")
    
    results_df.to_csv(output_file, index=False)
    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 80}")
    
    print("\nSummary Table:")
    print("-" * 80)
    print(f"{'Method':<20} {'N':<8} {'Avg Imp.':<12} {'Kappa Imp.':<12} {'Avg Flu.':<12} {'Kappa Flu.':<12}")
    print("-" * 80)
    for _, row in results_df.iterrows():
        avg_imp = row['avg_imperceptibility']
        kappa_imp = row['kappa_imperceptibility']
        avg_flu = row['avg_fluency']
        kappa_flu = row['kappa_fluency_weighted']
        
        if isinstance(avg_imp, str):
            avg_imp_str = f"{avg_imp:<12}"
        else:
            avg_imp_str = f"{avg_imp:<12.2f}"
        
        if isinstance(kappa_imp, str):
            kappa_imp_str = f"{kappa_imp:<12}"
        else:
            kappa_imp_str = f"{kappa_imp:<12.2f}"
        
        if isinstance(avg_flu, str):
            avg_flu_str = f"{avg_flu:<12}"
        else:
            avg_flu_str = f"{avg_flu:<12.2f}"
        
        if isinstance(kappa_flu, str):
            kappa_flu_str = f"{kappa_flu:<12}"
        else:
            kappa_flu_str = f"{kappa_flu:<12.2f}"
        
        print(f"{row['method']:<20} {int(row['num_documents']):<8} "
              f"{avg_imp_str} {kappa_imp_str} {avg_flu_str} {kappa_flu_str}")
    print("-" * 80)


if __name__ == "__main__":
    main()

