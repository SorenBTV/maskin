from NN import *
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import seaborn as sns

wine = load_wine()

inputs = wine.data
outputs = wine.target
outputs = outputs.reshape(outputs.shape[0], 1)

labels = wine.feature_names


X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#one hot encode outputs
onehot = OneHotEncoder(sparse=False)
y_train = onehot.fit_transform(y_train)
y_test = onehot.transform(y_test)


output_nodes = 3
input_nodes = X_train.shape[1]


intermediate_nodes = (5,)
dimensions = (input_nodes,  *intermediate_nodes, output_nodes)

#print(inputs.shape)
#print(labels)
my_network = FFNN(dimensions, output_func=softmax, cost_func=CostCrossEntropy, seed=123)

my_network.reset_weights() # reset weights such that previous runs or reruns don't affect the weights

#scheduler = Constant(eta=1e-3)
scheduler = Adam(eta=1e-2, rho=0.9, rho2=0.999)
scores = my_network.fit(X_train, y_train, scheduler, epochs=1000, X_val=X_test, t_val=y_test)

print("scores", scores.keys())

sns.lineplot(x=range(len(scores["train_accs"])), y=scores["train_accs"], label="Training accuracy")
sns.lineplot(x=range(len(scores["val_accs"])), y=scores["val_accs"], label="Test accuracy")
plt.show()
