import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm


def emb_model_train(embedded_df,estimators,lr,max_depth,obj,eval_met,ts_size,random_state):
    
    X = embedded_df.iloc[:, :384]  # Features (all columns except the last one)
    y = embedded_df.iloc[:, 384]  # Target variable (last column)

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=ts_size, 
                                                        random_state=random_state, 
                                                        stratify=y) # Stratified split
    
    # Initialize XGBoost classifier
    model = xgb.XGBClassifier(
        n_estimators=estimators, #1000,
        learning_rate=lr, #0.1,
        max_depth=max_depth,  #5,
        # Add other hyperparameters like:
        # subsample=0.8,  # Fraction of samples used for each tree
        # colsample_bytree=0.8, # Fraction of features used for each tree
        # gamma=0.1,         # Minimum loss reduction required to make a further partition on a leaf node
        # reg_alpha=1,       # L1 regularization term on weights
        # reg_lambda=1,      # L2 regularization term on weights
        objective=obj, #'binary:logistic',  # Explicitly set for binary classification
        use_label_encoder=False,
        eval_metric= eval_met #'logloss'  # Or 'auc', 'error', 'binary:logistic'
    )

    # Train with early stopping
    model.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=True)
    
    return model

def predict(model, X_test, y_test):
    # Predict target values
    y_pred = model.predict(X_test)
    # Predict probabilities
    y_pred_prob = model.predict_proba(X_test)[:,1]
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # Print accuracy
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    # Print classification report
    print(classification_report(y_test, y_pred))
    # Print confusion matrix
    print(confusion_matrix(y_test, y_pred))
    xgb.plot_importance(model)
    # Return the accuracy
    return y_pred, y_pred_prob

def make_emb_valid(valid_set_x):
    for i in tqdm(range(0,len(valid_set_x))):
        if i == 0:
            valid_ts = valid_set_x[i].head(1)
        elif i != 0 and i < len(valid_set_x):
            valid_ts = pd.concat([valid_ts,valid_set_x[i].head(1)],axis=0)
        else:
            valid_ts = pd.concat([valid_ts,valid_set_x[i]],axis=0)



