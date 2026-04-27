from src.training import train_and_save_model


if __name__ == '__main__':
    _, summary = train_and_save_model()
    print('Model training completed successfully.')
    print(f'Accuracy:  {summary.accuracy:.4f}')
    print(f'Precision: {summary.precision:.4f}')
    print(f'Recall:    {summary.recall:.4f}')
    print(f'F1-score:  {summary.f1_score:.4f}')
    print(f'ROC-AUC:   {summary.roc_auc:.4f}')
    print(f'Train size: {summary.train_size}')
    print(f'Test size:  {summary.test_size}')
