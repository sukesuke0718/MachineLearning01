//////////////////////////////////////////////////////////////////////
// ニューラルネットワーク全体を表現するためのクラス。
// void AddLayer(int numberOfNodes)
// void CalcForward(double[] inputdata)
// int GetMaxOutput()
// void InitWeight()
// void UpdateWeight(double alpha)
// void CalcError(double[] trainData)
// void SaveWeight()
// void LoadWeight()
// から構成される
//////////////////////////////////////////////////////////////////////
using System.IO;
using System.Collections.Generic;

public class NeruralNetwork
{

    public List<Layer> layers = new List<Layer>();

    //層を作成し、隣の層と全結合させる関数
    public void AddLayer(int numberOfNodes)
    {
        Layer newlayer = new Layer(numberOfNodes);

        if (layers.Count > 0)
        {
            layers[layers.Count - 1].ConnectionDensely(newlayer);
        }

        layers.Add(newlayer);

    }

    // 入力層にデータを入力後、出力層に向けて計算を行う関数
    public void CalcForward(double[] inputdata)
    {
        layers[0].SetInputData(inputdata);          // 入力層にデータを入力

        for (int i = 0; i < layers.Count; i++)
        {
            layers[i].CalcForward();
        }

    }

    // 出力層から認識結果を取得する関数
    public int GetMaxOutput()
    {
        int max = 0;
        double maxVule = 0.0;

        // 出力層のノードの中から最大値を見つける
        for (int i = 0; i < layers[layers.Count - 1].nodes.Count; i++)
        {
            if (layers[layers.Count - 1].nodes[i].value > maxVule)
            {
                max = i;        //最大値を持つノードのインデックスを記録
                maxVule = layers[layers.Count - 1].nodes[i].value;
            }

        }

        return max;     // 最大値を持つノードのインデックスを返す

    }

    public void InitWeight()
    {
        foreach (Layer layer in layers)
        {
            layer.InitWeight();
        }
    }

    /*
    CalcErrorは出力層の誤差を計算した後、隠れ層の誤差を計算する関数。
    UpdateWeightはエッジの重みを更新する関数。
    これら二つを使用することで誤差逆伝搬法を表現している
    */
    public void UpdateWeight(double alpha)
    {
        foreach (Layer layer in layers)
        {
            layer.UpdateWeight(alpha);
        }
    }

    public void CalcError(double[] trainData)
    {
        // 出力層のノード数を取得
        int numOutput = layers[layers.Count - 1].nodes.Count;
        // 出力層のノードの誤差を計算
        for (int i = 0; i < numOutput; i++)
        {
            layers[layers.Count - 1].nodes[i].CalcError(trainData[i]);
        }
        // 隠れ層のノードの誤差を計算
        for (int i = layers.Count - 2; i >= 0; i--)
        {
            foreach (Node node in layers[i].nodes)
            {
                node.CalcError();
            }
        }
    }

    // ニューラルネットワークの構成と全体の重みをファイル
    // 「weight.dat」に保存する関数
    public void SaveWight()
    {
        // ファイルを作成
        using (FileStream fsWeights =
            new FileStream(@"weight.dat", FileMode.OpenOrCreate))
        {
            using (BinaryWriter bwWeights =
                new BinaryWriter(fsWeights))
            {
                bwWeights.Write(layers.Count);      // 層の数を書き込む
                foreach (Layer layer in layers)
                {
                    // 各層のノード数を書き込む
                    bwWeights.Write(layer.nodes.Count);
                }

                foreach (Layer layer in layers)
                {
                    foreach (Node node in layer.nodes)
                    {
                        foreach (Edge edge in node.inputs)
                        {
                            // 重みを書き込む
                            bwWeights.Write(edge.weight);
                        }
                    }
                }

            }

        }

    }

    // ファイル「weight.dat」を読み込んで
    // ニューラルネットワークを作成し、全体の重みを設定する関数
    public void LoadWeight()
    {
        // ファイルを開く
        using (FileStream fsWeighws =
            new FileStream(@"weight.dat", FileMode.Open))
        {
            using (BinaryReader brWeights =
                new BinaryReader(fsWeighws))
            {
                layers = new List<Layer>();
                // 層の数を読み込む
                int numLayers = brWeights.ReadInt32();
                for (int i = 0; i < numLayers; i++)
                {
                    // 各層のノード数を読み込む
                    int numNodes = brWeights.ReadInt32();
                    AddLayer(numNodes);         // 層を作成する
                }

                /* この時点でニューラルネットワークが作成される */

                foreach (Layer layer in layers)
                {
                    foreach (Node node in layer.nodes)
                    {
                        foreach (Edge edge in node.inputs)
                        {
                            // 重みを読み込んで設定する
                            edge.weight = brWeights.ReadDouble();
                        }
                    }
                }
            }
        }
    }
}