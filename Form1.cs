//////////////////////////////////////////////////////////////////////
// フォーム
//////////////////////////////////////////////////////////////////////

using System;
using System.Drawing;
using System.IO;
using System.Threading;
using System.Windows.Forms;

namespace MachineLearning
{
    public partial class Form1 : Form
    {
        NeruralNetwork nn;
        const int num_input_nodes = 28 * 28;        // 入力層のノード数
        const int num_hidden_nodes = 100;           // 隠れ層のノード数
        const int num_output_nodes = 10;            // 出力層のノード数
        double alpha = 0.1;                         // 学習率

        // 6万個のデータのうち5万個を学習用、1万個を検証用に使用する
        const int num_training_data = 50000;
        const int num_test_data = 10000;
        const int num_data = num_training_data + num_test_data;

        double[][] pixcel;                          // 画像のピクセルデータ
        double[][] label;                           // 出力層の誤差計算用の正解（ラベル）データ
        int[] labelIndex;                           // 正解（ラベル）データ
        int px0;
        int py0;

        // コンストラクタ
        public Form1()
        {
            InitializeComponent();

            // 3つの層のあるニューラルネットワークを構築する
            nn = new NeruralNetwork();
            nn.AddLayer(num_input_nodes);
            nn.AddLayer(num_hidden_nodes);
            nn.AddLayer(num_output_nodes);
        }


        // ボタンが押されたときに実行される関数（イベントハンドラ）
        private void button1_Click(object sender, EventArgs e)
        {
            // 別スレッドで機械学習を開始
            Thread t = new Thread(new ThreadStart(train));
            t.Start();
            buttonEnable(false);
        }

        // ボタンの有効化・無効化を行う処理
        delegate void buttonEnableDelegate(bool b);
        void buttonEnable(bool b)
        {
            if (InvokeRequired)
            {
                Invoke(new buttonEnableDelegate(buttonEnable), new object[] { b });
                return;
            }
            button1.Enabled = b;
        }

        // テキストを表示する処理
        delegate void SetTextDelegate(string s);
        void setText1(string s)
        {
            if (InvokeRequired)
                Invoke(new SetTextDelegate(setText1), new object[] { s });
            else
                label1.Text = s;
        }

        // 機械学習の関数
        private void train()
        {
            GraphInit();
            loadData();
            nn.InitWeight();        // 重みを初期化

            // 学習データの数だけ機械学習を行うループ
            for (int i = 0; i < num_training_data; i++)
            {
                nn.CalcForward(pixcel[i]);
                nn.CalcError(label[i]);
                nn.UpdateWeight(alpha);

                // 100コごとに精度を検証して表示する
                if (i % 100 == 0)
                {
                    Test(100, i);
                    setText1(i.ToString());
                }
            }

            // 機械学習完了後に最終的な精度を表示する
            setText1("Testing...");
            double score = Test(num_test_data, num_training_data);
            setText1((score * 100.0).ToString("F2") + "%");
            // weight.datとlabel.txtの作成(結果の保存)
            DialogResult result = MessageBox.Show("学習結果を保存しますか？",
                "Save", MessageBoxButtons.YesNo, MessageBoxIcon.Question);
            if(result == DialogResult.Yes)
            {
                nn.SaveWight();
                SaveLabelName(new string[] { "0", "1", "2", "3", "4", "5", "6","7", "8", "9"});
            }
            buttonEnable(true);
        }

        // 精度を検証する関数
        double Test(int dataLength, int graphX)
        {
            int ok = 0;
            int offset = num_training_data;
            int lastData = offset + dataLength;
            for(int i = 0; i < dataLength; i++)
            {
                nn.CalcForward(pixcel[offset + i]);
                if (nn.GetMaxOutput() == labelIndex[offset + i])
                {
                    ok++;
                }
            }

            double score = (double)ok / (double)dataLength;
            GraphPlot((double)graphX / (double)num_training_data, score);

            return score;
        }

        // MNISTのファイルを読み込む関数
        public void loadData()
        {

            pixcel = new double[num_data][];
            label = new double[num_data][];
            labelIndex = new int[num_data];

            using (FileStream fsPixcel =
                new FileStream(@"train-images.idx3-ubyte", FileMode.Open))
            {
                using (BinaryReader brPixcel = new BinaryReader(fsPixcel))
                {
                    brPixcel.ReadInt32();      // ヘッダーを読み飛ばす
                    brPixcel.ReadInt32();
                    brPixcel.ReadInt32();
                    brPixcel.ReadInt32();

                    // ピクセル（手書き数字の画像）を読み込む
                    for(int i = 0; i < num_data; i++)
                    {
                        pixcel[i] = new double[num_input_nodes];
                        for(int j = 0; j < num_input_nodes; j++)
                        {
                            pixcel[i][j] = (double)brPixcel.ReadByte() / 255.0 * 0.99 + 0.01;
                        }
                    }
                }
            }

            using (FileStream fsLabel =
                new FileStream(@"train-labels.idx1-ubyte", FileMode.Open))
            {
                using (BinaryReader brLabel = new BinaryReader(fsLabel))
                {
                    brLabel.ReadInt32();      // ヘッダーを読み飛ばす
                    brLabel.ReadInt32();

                    // 正解データを読み込む
                    for(int i = 0; i < num_data; i++)
                    {
                        label[i] = new double[num_output_nodes];
                        for(int j = 0; j < num_output_nodes; j++)
                        {
                            label[i][j] = 0.01;
                        }

                        labelIndex[i] = brLabel.ReadByte();
                        label[i][labelIndex[i]] = 0.99;

                    }

                }

            }
        }

        // 正解の種類をファイルに保存する関数
        private void SaveLabelName(string[] labelNames)
        {
            using (StreamWriter writer = new StreamWriter(@"label.txt"))
            {
                for(int i = 0; i < labelNames.Length; i++)
                {
                    if (i > 0)
                        writer.Write(",");

                    writer.Write(labelNames[i]);
                }
            }

        }

        // ピクチャーボックスを初期化する関数
        private void GraphInit()
        {
            px0 = -1;
            py0 = -1;
            Graphics g = pictureBox1.CreateGraphics();
            g.Clear(Color.White);
            g.Dispose();
        }

        // ピクチャーボックスにグラフを描く関数
        private void GraphPlot(double x, double y)
        {

            if(px0 < 0)
            {
                px0 = 0;
                py0 = pictureBox1.Height;
            }

            x = Math.Min(1.0, Math.Max(0.0, x));
            x = pictureBox1.Width * x;
            y = Math.Min(1.0, Math.Max(0.0, y));
            y = pictureBox1.Height * (1.0 - y);

            int px1 = (int)x;
            int py1 = (int)y;

            Graphics g = pictureBox1.CreateGraphics();
            g.DrawLine(Pens.Blue, px0, py0, px1, py1);
            px0 = px1;
            py0 = py1;
            g.Dispose();
        }

    }
}
