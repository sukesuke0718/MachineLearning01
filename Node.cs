//////////////////////////////////////////////////////////////////////
// ニューラルネットワークの構成要素における「ノード」及び「エッジ」のクラスを
// 所持している。
// ノードでは入力値の合計を活性化関数に与え、出力値として変換を行っている。
// また、活性化関数としては「シグモレイド関数」を使用することとする。
// 「エッジ」は各層をつなぐ役割をしている。
//////////////////////////////////////////////////////////////////////
using System;
using System.Collections.Generic;

// エッジを表すクラス
public class Edge
{
    public Node left;            // 入力側
    public Node right;           // 出力側
    public double weight;        // 重み
}

// ノードを表すクラス
public class Node
{

    public List<Edge> inputs = new List<Edge>();
    public List<Edge> outputs = new List<Edge>();
    public double inValue;                              // 入出力の合計
    public double value;                                // 出力値
    public double error;                                // 誤差
    static Random random = new Random();

    // 活性化関数
    public double Activation(double val)
    {
        return 1.0 / (1.0 + Math.Exp(-val));
    }

    // 活性化関数を微分した関数
    public double DActivation(double val)
    {
        return (1.0 - val) * val;
    }

    // 隣のノードと接続する関数
    public Edge Connect(Node right)
    {

        Edge edge = new Edge();
        edge.left = this;
        edge.right = right;
        right.inputs.Add(edge);
        this.outputs.Add(edge);

        return edge;
    }

    // 出力値を計算する関数
    public void CalcForward()
    {

        if (inputs.Count == 0)
        {
            return;
        }

        inValue = 0.0;

        // foreach は配列の要素をひとつずつ読み出す構文
        foreach (Edge edge in inputs)
        {
            inValue += edge.left.value * edge.weight;
        }
        value = Activation(inValue);
    }

    // 正規分布乱数を生成する関数
    public static double GetRandom()
    {

        double r1 = random.NextDouble();
        double r2 = random.NextDouble();

        // ボックス・ミュラー法
        return (Math.Sqrt(-2.0 * Math.Log(r1)) *
                    Math.Cos(2.0 * Math.PI * r2)) * 0.1;

    }

    // 重みを乱数で初期化する関数
    public void InitWeight()
    {

        foreach (Edge edge in inputs)
        {
            edge.weight = GetRandom();
        }

    }

    // （手順１）誤差計算（出力層）の関数
    public void CalcError(double t)
    {
        error = t - value;
    }

    // （手順２）誤差計算（隠れ層）の関数
    public void CalcError()
    {

        error = 0.0;
        foreach (Edge edge in outputs)
        {
            error += edge.weight * edge.right.error;
        }

    }

    // （手順３）重み更新関数
    public void UpdateWeight(double alpha)      // alpha：学習率
    {

        foreach (Edge edge in inputs)
        {
            // 調整値の算出
            // 誤差が大きいほど調整値も大きくなる
            edge.weight += alpha * error *
                DActivation(value) * edge.left.value;
        }
    }

}