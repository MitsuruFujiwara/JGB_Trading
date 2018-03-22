# JGB_Trading
Deep Neural Network(DNN)を使い、10日後のJGB利回りの変化を予測する。  
入力値として直近50日間のJGB利回りの推移と季節性のフラグを用いる。  
元データは財務省HPで公開されているものを使用。  
http://www.mof.go.jp/jgbs/reference/interest_rate/index.htm  

JGB利回りのデータの取得にあたって以下のモジュールを使用。  
https://github.com/MitsuruFujiwara/JGBScraping  

メインの処理などはTrading.ipynbを参照。
