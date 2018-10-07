//////////////////////////////////////////////////////////////////////
// ニューラルネットワークの構成要素における「層」を表すクラスを
// 所持している。
// 「入力層」「隠れ層」「出力層」を表現する
//////////////////////////////////////////////////////////////////////
using System;
using System.Collections.Generic;

public class Layer
{

    public List<Node> nodes = new List<Node>();

    // コンストラクタ
    public Layer(int numNodes)
    {
        for (int i = 0; i < numNodes; i++)
        {
            Node node = new Node();
            nodes.Add(node);
        }
    }

    // 自分のノードと隣のレイヤーのノード接続するためのもの
    // ノードの全結合のための関数
    public void ConnectionDensely(Layer rightLayer)
    {
        foreach (Node node in nodes)
        {
            foreach (Node nextNode in rightLayer.nodes)
            {
                node.Connect(nextNode);
            }
        }
    }

    public void InitWeight()
    {
        foreach (Node node in nodes)
        {
            node.InitWeight();
        }

    }

    // ノードにデータを入力するための関数
    public void SetInputData(double[] input)
    {
        for (int i = 0; i < nodes.Count; i++)
        {
            nodes[i].value = input[i];
        }

    }

    public void CalcForward()
    {
        foreach (Node node in nodes)
        {
            node.CalcForward();
        }
    }

    public void UpdateWeight(double alpha)
    {
        foreach (Node node in nodes)
        {
            node.UpdateWeight(alpha);
        }
    }


}