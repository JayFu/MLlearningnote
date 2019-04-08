// 
#include<iostream>  
#include<cmath>
#include<time.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
using namespace std;  
  
#define  innode 3  //输入结点数  
#define  hidenode 10//隐含结点数  
#define  outnode 1 //输出结点数  
#define  trainsample 8//BP训练样本数  
class BpNet  
{  
public:  
    //BP train
    void train(double p[trainsample][innode ],double t[trainsample][outnode]);  
    // input
    double p[trainsample][innode];     //输入的样本  
    // output
    double t[trainsample][outnode];    //样本要输出的  
    // bp recognize
    double *recognize(double *p);//Bp识别  

    // write weight
    void writetrain(); //写训练完的权值  
    // read weight
    void readtrain(); //读训练好的权值，这使的不用每次去训练了，只要把训练最好的权值存下来就OK  
  
    BpNet();  
    virtual ~BpNet();  
  
public:  
    void init();  
    // hide node weight
    double w[innode][hidenode];//隐含结点权值  
    // output node weight 
    double w1[hidenode][outnode];//输出结点权值  
    // hide node threshold
    double b1[hidenode];//隐含结点阀值  
    // output node threshold
    double b2[outnode];//输出结点阀值  

    // weight learning rate(input to hide)
    double rate_w; //权值学习率（输入层-隐含层)  
    // weight learning rate(hide to output)
    double rate_w1;//权值学习率 (隐含层-输出层)  
    // threshold learning rate hide
    double rate_b1;//隐含层阀值学习率  
    // threshold learning rate output
    double rate_b2;//输出层阀值学习率  

    // error
    double e;//误差计算  
    // max error
    double error;//允许的最大误差
    // bp output  
    double result[outnode];// Bp输出  
};  
  
BpNet::BpNet()  
{  
    error=1.0;  
    e=0.0;  

    // input to hiden learning rate
    rate_w=0.9;  //权值学习率（输入层--隐含层)
    // hiden to output learning rate
    rate_w1=0.9; //权值学习率 (隐含层--输出层)
    // hiden bias learning rate
    rate_b1=0.9; //隐含层阀值学习率
    // output bias learning rate
    rate_b2=0.9; //输出层阀值学习率
}

BpNet::~BpNet()
{

}

// init weight
void winit(double w[],int n) //权值初始化
{
  for(int i=0;i<n;i++)
    w[i]=(2.0*(double)rand()/RAND_MAX)-1;
}

void BpNet::init()
{
    winit((double*)w,innode*hidenode);
    winit((double*)w1,hidenode*outnode);
    winit(b1,hidenode);
    winit(b2,outnode);
}  
  
void BpNet::train(double p[trainsample][innode],double t[trainsample][outnode])  
{
    // hiden node error
    double pp[hidenode];//隐含结点的校正误差
    // deviation
    double qq[outnode];//希望输出值与实际输出值的偏差
    // ideal output
    double yd[outnode];//希望输出值  

    // input
    double x[innode]; //输入向量
    // hiden node status value
    double x1[hidenode];//隐含结点状态值
    // output node status value
    double x2[outnode];//输出结点状态值
    // hiden activate value
    double o1[hidenode];//隐含层激活值
    // output activate value
    double o2[hidenode];//输出层激活值

    // for each input
    for(int isamp=0;isamp<trainsample;isamp++)//循环训练一次样品
    {
        for(int i=0;i<innode;i++)
            // input
            x[i]=p[isamp][i]; //输入的样本
        for(int i=0;i<outnode;i++)
            // ideal output
            yd[i]=t[isamp][i]; //期望输出的样本
        
        // #pragma omp parallel for
        // instruct standar input and output for each data
        //构造每个样品的输入和输出标准
        for(int j=0;j<hidenode;j++)
        {
            o1[j]=0.0;
            for(int i=0;i<innode;i++)
                // activate value for hiden node
                o1[j]=o1[j]+w[i][j]*x[i];//隐含层各单元输入激活值
            // output of hiden node
            x1[j]=1.0/(1+exp(-o1[j]-b1[j]));//隐含层各单元的输出
        }

        for(int k=0;k<outnode;k++)
        {
            o2[k]=0.0;
            // #pragma omp parallel for
            for(int j=0;j<hidenode;j++)
                // activate value for output node
                o2[k]=o2[k]+w1[j][k]*x1[j]; //输出层各单元输入激活值
            // output of output node
            x2[k]=1.0/(1.0+exp(-o2[k]-b2[k])); //输出层各单元输出
        }  

        for(int k=0;k<outnode;k++)
        {   
            // deviation between output and ideal output
            qq[k]=(yd[k]-x2[k])*x2[k]*(1-x2[k]); //希望输出与实际输出的偏差
            for(int j=0;j<hidenode;j++)
                // weight between hiden and output next time
                w1[j][k]+=rate_w1*qq[k]*x1[j];  //下一次的隐含层和输出层之间的新连接权
        }

        for(int j=0;j<hidenode;j++)
        {
            pp[j]=0.0;
            for(int k=0;k<outnode;k++)
                pp[j]=pp[j]+qq[k]*w1[j][k];
            // error of hiden node
            pp[j]=pp[j]*x1[j]*(1-x1[j]); //隐含层的校正误差

            for(int i=0;i<innode;i++)
                // weight between hiden and input next time
                w[i][j]+=rate_w*pp[j]*x[i]; //下一次的输入层和隐含层之间的新连接权
        }

        for(int k=0;k<outnode;k++)
        {   
            // mean square deviation
            e+=fabs(yd[k]-x2[k])*fabs(yd[k]-x2[k]); //计算均方差
        }
        error=e/2.0;

        for(int k=0;k<outnode;k++)
            // bias between hiden layer and output layer next time
            b2[k]=b2[k]+rate_b2*qq[k]; //下一次的隐含层和输出层之间的新阈值
        
        // #pragma omp parallel for
        for(int j=0;j<hidenode;j++)
            // bias between hiden layer and input layer next time
            b1[j]=b1[j]+rate_b1*pp[j]; //下一次的输入层和隐含层之间的新阈值  
    }
}

double *BpNet::recognize(double *p)  
{  
    double x[innode]; //输入向量  
    double x1[hidenode]; //隐含结点状态值  
    double x2[outnode]; //输出结点状态值  
    double o1[hidenode]; //隐含层激活值  
    double o2[hidenode]; //输出层激活值  
  
    for(int i=0;i<innode;i++)  
        x[i]=p[i];  
  
    for(int j=0;j<hidenode;j++)  
    {  
        o1[j]=0.0;  
        for(int i=0;i<innode;i++)  
            o1[j]=o1[j]+w[i][j]*x[i]; //隐含层各单元激活值  
        x1[j]=1.0/(1.0+exp(-o1[j]-b1[j])); //隐含层各单元输出  
        //if(o1[j]+b1[j]>0) x1[j]=1;  
        //    else x1[j]=0;  
    }  
  
    for(int k=0;k<outnode;k++)  
    {  
        o2[k]=0.0;  
        for(int j=0;j<hidenode;j++)  
            o2[k]=o2[k]+w1[j][k]*x1[j];//输出层各单元激活值  
        x2[k]=1.0/(1.0+exp(-o2[k]-b2[k]));//输出层各单元输出  
        //if(o2[k]+b2[k]>0) x2[k]=1;  
        //else x2[k]=0;  
    }  
  
    for(int k=0;k<outnode;k++)  
    {  
        result[k]=x2[k];  
    }  
    return result;  
}  
  
void BpNet::writetrain()  
{  
    FILE *stream0;  
    FILE *stream1;  
    FILE *stream2;  
    FILE *stream3;  
    int i,j;  
    //隐含结点权值写入  
    if(( stream0 = fopen("w.txt", "w+" ))==NULL)  
    {  
        cout<<"创建文件失败!";  
        exit(1);  
    }  
    for(i=0;i<innode;i++)  
    {  
        for(j=0;j<hidenode;j++)  
        {  
            fprintf(stream0, "%f\n", w[i][j]);  
        }  
    }  
    fclose(stream0);  
  
    //输出结点权值写入  
    if(( stream1 = fopen("w1.txt", "w+" ))==NULL)  
    {  
        cout<<"创建文件失败!";  
        exit(1);  
    }  
    for(i=0;i<hidenode;i++)  
    {  
        for(j=0;j<outnode;j++)  
        {  
            fprintf(stream1, "%f\n",w1[i][j]);  
        }  
    }  
    fclose(stream1);  
  
    //隐含结点阀值写入  
    if(( stream2 = fopen("b1.txt", "w+" ))==NULL)  
    {  
        cout<<"创建文件失败!";  
        exit(1);  
    }  
    for(i=0;i<hidenode;i++)  
        fprintf(stream2, "%f\n",b1[i]);  
    fclose(stream2);  
  
    //输出结点阀值写入  
    if(( stream3 = fopen("b2.txt", "w+" ))==NULL)  
    {  
        cout<<"创建文件失败!";  
        exit(1);  
    }  
    for(i=0;i<outnode;i++)  
        fprintf(stream3, "%f\n",b2[i]);  
    fclose(stream3);  
  
}  
  
void BpNet::readtrain()  
{  
    FILE *stream0;  
    FILE *stream1;  
    FILE *stream2;  
    FILE *stream3;  
    int i,j;  
  
    //隐含结点权值读出  
    if(( stream0 = fopen("w.txt", "r" ))==NULL)  
    {  
        cout<<"打开文件失败!";  
        exit(1);  
    }  
    float  wx[innode][hidenode];  
    for(i=0;i<innode;i++)  
    {  
        for(j=0;j<hidenode;j++)  
        {  
            fscanf(stream0, "%f", &wx[i][j]);  
            w[i][j]=wx[i][j];  
        }  
    }  
    fclose(stream0);  
  
    //输出结点权值读出  
    if(( stream1 = fopen("w1.txt", "r" ))==NULL)  
    {  
        cout<<"打开文件失败!";  
        exit(1);  
    }  
    float  wx1[hidenode][outnode];  
    for(i=0;i<hidenode;i++)  
    {  
        for(j=0;j<outnode;j++)  
        {  
            fscanf(stream1, "%f", &wx1[i][j]);  
            w1[i][j]=wx1[i][j];  
        }  
    }  
    fclose(stream1);  
  
    //隐含结点阀值读出  
    if(( stream2 = fopen("b1.txt", "r" ))==NULL)  
    {  
        cout<<"打开文件失败!";  
        exit(1);  
    }  
    float xb1[hidenode];  
    for(i=0;i<hidenode;i++)  
    {  
        fscanf(stream2, "%f",&xb1[i]);  
        b1[i]=xb1[i];  
    }  
    fclose(stream2);  
  
    //输出结点阀值读出  
    if(( stream3 = fopen("b2.txt", "r" ))==NULL)  
    {  
        cout<<"打开文件失败!";  
        exit(1);  
    }  
    float xb2[outnode];  
    for(i=0;i<outnode;i++)  
    {  
        fscanf(stream3, "%f",&xb2[i]);  
        b2[i]=xb2[i];  
    }  
    fclose(stream3);  
}  
  


int main()  
{   
    // input and ideal output
    // Y is lable vector, X is data vector

    time_t startt, stopt;
    startt = time(NULL);
    BpNet bp;  
    bp.init();
    int data_count;
    for(int ccount = 1; ccount < 500000; ccount++)  
    {   
        data_count = ccount - 1460 * (ccount / 1460); 
        bp.e=0.0;    
        bp.train(X[data_count],Y[data_count]);  
        if (ccount % 100000 == 0)
            cout<<"Times="<<ccount<<" error="<<bp.error<<endl;  
    }  
    cout<<"trainning complete..."<<endl; 
    // double m[innode]={1,1,1};  
    // double *r=bp.recognize(m);  
    // for(int i=0;i<outnode;++i)  
    //    cout<<bp.result[i]<<" ";  
    // double cha[trainsample][outnode];  
    // double mi=100;  
    // double index;  
    // for(int i=0;i<trainsample;i++)  
    // {  
    //     for(int j=0;j<outnode;j++)  
    //     {  
    //         //找差值最小的那个样本  
    //         cha[i][j]=(double)(fabs(Y[i][j]-bp.result[j]));  
    //         if(cha[i][j]<mi)  
    //         {  
    //             mi=cha[i][j];  
    //             index=i;  
    //         }  
    //     }  
    // }  
    // for(int i=0;i<innode;++i)  
    //    cout<<m[i];  
    // cout<<" is "<<index<<endl;  
    // cout<<endl;
    stopt = time(NULL);
    cout<<"time cost: "<<(stopt - startt)<<endl;  
    return 0;  
}
