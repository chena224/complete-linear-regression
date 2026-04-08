import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split,cross_validate,GridSearchCV,learning_curve
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.inspection import permutation_importance

import warnings
warnings.filterwarnings('ignore')

# 中文字体配置（保证可视化中文正常显示）
from matplotlib import font_manager
chinese_fonts = [f.name for f in font_manager.fontManager.ttflist if any(k in f.name.lower() for k in ['hei', 'song', 'yahei'])]
plt.rcParams["font.family"] = chinese_fonts[0] if chinese_fonts else "sans-serif"
plt.rcParams["axes.unicode_minus"] = False


np.set_printoptions(threshold=np.inf,linewidth=np.inf)
# 使用 np.inf 设置 pandas 显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

class CompleteLinearRegression:
    def __init__(self,test_size=0.1,cvfolds=5,random_state=43):
        self.test_size=test_size
        self.cvfolds=cvfolds
        self.random_state=random_state
        self.best_model=None
        self.best_params=None
        self.results=None

    def auto_fit(self,data,X,y,model_type='Ridge',tunehyperparams=True):
        self.data=data
        self.X=X
        self.y=y
        self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(X,y,test_size=self.test_size,random_state=self.random_state)

        self.pipeline=Pipeline([
            ('scale',StandardScaler()),
            ('poly',PolynomialFeatures(degree=1,include_bias=False)),
            ('model',None)
        ])

        model_configs={
            'LinearRegression':{
                'model':LinearRegression(),
                'params':{
                    'poly__degree':[1,2,3,4],
                }
            },
            'Ridge':{
                'model':Ridge(alpha=0.01,random_state=self.random_state),
                'params':{
                    'poly__degree':[1,2,3,4],
                    'model__alpha':[0.001,0.01,0.1,1.0,10.0,100.0]
                }
            },
            'Lasso':{
                'model':Lasso(alpha=0.01,random_state=self.random_state),
                'params':{
                    'poly__degree':[1,2,3,4],
                    'model__alpha':[0.001,0.01,0.1,1.0,10.0,100.0]
                }
            },
            'ElasticNet':{
                    'model':ElasticNet(alpha=0.01,random_state=self.random_state),
                    'params':{
                        'poly__degree':[1,2,3,4],
                        'model__alpha':[0.001,0.01,0.1,1.0,10.0,100.0]
                }
            }
        }
        config=model_configs[model_type]
        self.pipeline.set_params(model=config['model'])

        if tunehyperparams:
            print("🔍 正在进行超参数搜索...")
            search=GridSearchCV(self.pipeline,config['params'],cv=self.cvfolds,scoring='r2',verbose=1)
            search.fit(self.X_train,self.y_train)

            self.best_model=search.best_estimator_
            self.best_params=search.best_params_

            print(f'超参数搜索后选取的最优参数:{self.best_params}')

        else:
            self.best_model=self.pipeline
            self.best_model.fit(self.X_train,self.y_train)
            if model_type=='LinearRegression':
               print(f'由于未进行超参数搜索,我们使用最初设定参数:degree:2')
            else:
                print(f'由于未进行超参数搜索,我们使用最初设定参数:degree:2,alpha:0.01')
        
        metrics=self.evaluate()

        return metrics
    
    def evaluate(self):
        y_pred_train=self.best_model.predict(self.X_train)
        y_pred_test=self.best_model.predict(self.X_test)

        metrics={
            'train_r2':r2_score(self.y_train,y_pred_train),
            'train_mae':mean_absolute_error(self.y_train,y_pred_train),
            'train_mse':mean_squared_error(self.y_train,y_pred_train),
            'test_r2':r2_score(self.y_test,y_pred_test),
            'test_mae':mean_absolute_error(self.y_test,y_pred_test),
            'test_mse':mean_squared_error(self.y_test,y_pred_test)
        }

        cross_score=cross_validate(self.best_model,self.X_train,self.y_train,cv=self.cvfolds,scoring='r2')
        metrics['cv_r2']=cross_score['test_score'].mean()

        self.results={
            'metrics':metrics,
            'prediction':y_pred_test,
            'feature_importance':None
        }

        return metrics
    
    def plot_learning_curve(self):
        train_size,train_score,test_score=learning_curve(self.best_model,self.X_train,self.y_train,cv=self.cvfolds,scoring='neg_mean_squared_error',train_sizes=np.linspace(0.1,1,10),random_state=self.random_state)
        train_score_mean=-train_score.mean(axis=1)
        test_score_mean=-test_score.mean(axis=1)

        plt.figure(figsize=(12,8))
        plt.plot(train_size,train_score_mean,c='r',label='训练集分数')
        plt.plot(train_size,test_score_mean,c='k',label='测试集分数')
        plt.legend(loc='best')
        plt.xlabel('训练样本数量')
        plt.ylabel('均方误差')
        plt.title('学习曲线')
        plt.grid(True,alpha=0.3)
        plt.show()

    def plot_features_importance(self,top_k=8):
        importance=permutation_importance(self.best_model,self.X_test,self.y_test,n_repeats=6,random_state=self.random_state)
        feature_names=self.data.columns.drop(self.y.name)

        importance_df=pd.DataFrame({
            'features':feature_names,
            'importance':importance.importances_mean,
        }).sort_values('importance',ascending=True).tail(top_k)
        x_pos=np.arange(len(importance_df))
        plt.figure(figsize=(12,8))
        plt.bar(x_pos,importance_df['importance'],color='purple')
        plt.xticks(x_pos, importance_df['features'])
        plt.xlabel('特征重要性得分')
        plt.title(f'Top {top_k} 特征重要性')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.show()

        # 保存结果
        self.results['feature_importance'] = importance_df
        
        return importance_df
 

        
    def get_detailed_summary(self):
        """获取详细模型摘要"""
        metrics = self.results.get('metrics', {})
        
        print("\n" + "="*50)
        print("           完整模型训练摘要")
        print("="*50)
        
        print(f"\n📊 性能指标:")
        print(f"  训练集 R²: {metrics.get('train_r2', 0):.4f}")
        print(f"  测试集 R²: {metrics.get('test_r2', 0):.4f}")
        print(f"  训练集 MAE: {metrics.get('train_mae', 0):.4f}")
        print(f"  测试集 MAE: {metrics.get('test_mae', 0):.4f}")
        print(f"  训练集 MSE: {metrics.get('train_mse', 0):.4f}")
        print(f"  测试集 MSE: {metrics.get('test_mse', 0):.4f}")
        print(f"  交叉验证 R²: {metrics.get('cv_r2', 0):.4f}")
        
        print(f"\n⚙️  超参数配置:")
        print(f"  最佳参数: {self.best_params}")
        
        # 过拟合分析
        train_test_gap = metrics.get('train_r2', 0) - metrics.get('test_r2', 0)
        
        
        print(f"\n⚠️  过拟合分析:")
        print(f"  训练-测试集 R²差距: {train_test_gap:.4f}")
        if train_test_gap != 0:
            print(f"  交叉验证训练-测试集 R²差距: {train_test_gap:.4f}")
        
        # 性能评级
        test_r2 = metrics.get('test_r2', 0)
        if test_r2 >= 0.9:
            rating, emoji = "优秀", "🎯"
        elif test_r2 >= 0.7:
            rating, emoji = "良好", "👍"
        elif test_r2 >= 0.5:
            rating, emoji = "一般", "👌"
        else:
            rating, emoji = "需要改进", "⚠️"
        
        print(f"\n📈 总体评级: {rating} {emoji}")


def demo_complete() :
    data=pd.read_csv(r"D:\360MoveData\Users\不遵\Desktop\ex1data2.txt",header=None,names=['Size', 'Bedrooms', 'Price'])
    cols=data.shape[1]
    X=data.iloc[:,0:cols-1]
    y=data.iloc[:,cols-1]
    print("🚀 开始完整模型训练...")
    model = CompleteLinearRegression(cvfolds=2)
    metrics=model.auto_fit(data,X,y,model_type='LinearRegression',tunehyperparams=False)
    y_pred=model.best_model.predict(X)
    print(f'预测结果{y_pred}')
    # 查看详细摘要
    model.get_detailed_summary()
    
    print("\n📊 生成特征重要性图...")
    model.plot_features_importance(top_k=8)
    
    print("\n📈 生成学习曲线...")
    model.plot_learning_curve()
    return model,metrics
if __name__ == "__main__":
    model, metrics = demo_complete()



        

                





