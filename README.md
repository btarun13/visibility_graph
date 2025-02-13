# Visibility Graph

After reading several papers on graphs and time series, I came across an approach for utilizing time series data to construct graphs. These graphs are known as visibility graphs. As shown in the animation, points are connected based on the visibility of towers, which represent values at different time points. Using this method, we can generate graphs for different rolling windows, where each window represents a separate graph.



![Demonstration of the project's key features](https://github.com/btarun13/visibility_graph/blob/main/my_ts_animation.gif)



With this graph, we can derive vector representations that can be used for downstream predictions. This approach provides more information, not just the values at different time points but also spatiotemporal relationships.




![image for project](https://github.com/btarun13/visibility_graph/blob/main/combine_vg_window_10.png)


After generating graphs for each time series within different windows, I build embeddings using a GCN architecture. I use a one-layer aggregation, which corresponds to aggregating information from one hop. 

example.ipynb # for example

As a test for this approach, I used OHLC time series data for the stock daily. The target variable is based on the "close" price—specifically, whether the closing price increases within the next k days (k can be adjusted). The embeddings generated from the graph are then used as features for the target variable. An example of this process is provided in the example notebook. The dataset is in 7.csv, and it demonstrates how the data is formatted and passed through various functions to create representations for each window period.

For this dataset, an XGBoost model achieved an accuracy of 0.56. When applied to a simple long strategy, it yielded a cumulative return of 1.18 with a maximum drawdown of 0.50. While not a highly effective strategy on its own, this approach has the potential for further development. By integrating it with portfolio management frameworks, it could evolve into a more sophisticated method for analyzing time series data and developing trading strategies.

