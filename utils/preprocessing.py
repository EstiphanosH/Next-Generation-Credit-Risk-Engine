from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
def build_preprocessor(df, target_col):
    X=df.drop(columns=[target_col], errors="ignore")
    num=X.select_dtypes(include=["int64","float64","int32","float32"]).columns.tolist()
    cat=[c for c in X.columns if c not in num]
    num_pipe=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())])
    cat_pipe=Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore",sparse_output=False))])
    pre=ColumnTransformer([("num",num_pipe,num),("cat",cat_pipe,cat)]); return pre, num, cat
