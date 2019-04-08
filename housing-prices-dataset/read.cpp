#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
 
using namespace std;

int main()
{

	// 读文件
	ifstream inFile("label.csv", ios::in);
	string lineStr;
	vector<vector<string> > strArray;
	while (getline(inFile, lineStr))
	{
		// 存成二维表结构
		stringstream ss(lineStr);
		string str;
		vector<string> lineArray;
		// 按照逗号分隔
		while (getline(ss, str, ','))
			lineArray.push_back(str);
		strArray.push_back(lineArray);
	}
	getchar();
	return 0;
}


    int i;
    ifstream fin("label.csv");
    string label_line;
    vector<string> X[1460];
    vector<string> Y[1460];
    for(i=0;i<=1460;i++){
        getline(fin, label_line);
        if(i == 0) continue;
        Y[i-1].push_back(label_line);
    }
	ifstream din("train.csv"); 
	string line;
    for(i=0;i<=1460;i++){
        getline(din, line);
        if (i == 0) continue;
		X[i-1].push_back(line);
    }


















// int main()
// {   
//     int i;
//     ifstream fin("label.csv");
//     string label_line;
//     vector<string> X[1460];
//     vector<string> Y[1460];
//     for(i=0;i<=1460;i++){
//         getline(fin, label_line);
//         if(i == 0){
//             continue;
//         }
//         Y[i-1].push_back(label_line);
//     }

// 	ifstream din("train.csv"); //打开文件流操作
// 	string line;
//     for(i=0;i<=1460;i++){
//         getline(din, line);
//         if (i == 0)
//         {   
//             continue;
//         }
// 		X[i-1].push_back(line);
//         }



//     // }
// 	// while (getline(fin, line))   //整行读取，换行符“\n”区分，遇到文件尾标志eof终止读取
// 	// {   
// 	// 	cout <<"原始字符串："<< line << endl; //整行输出
// 	// 	istringstream sin(line); //将整行字符串line读入到字符串流istringstream中
// 	// 	vector<string> fields; //声明一个字符串向量
// 	// 	string field;
// 	// 	while (getline(sin, field, ',')) //将字符串流sin中的字符读入到field字符串中，以逗号为分隔符
// 	// 	{
// 	// 		fields.push_back(field); //将刚刚读取的字符串添加到向量fields中
// 		// }

// 		// string name = Trim(fields[0]); //清除掉向量fields中第一个元素的无效字符，并赋值给变量name
// 		// string age = Trim(fields[1]); //清除掉向量fields中第二个元素的无效字符，并赋值给变量age
// 		// string birthday = Trim(fields[2]); //清除掉向量fields中第三个元素的无效字符，并赋值给变量birthday
// 		// cout <<"处理之后的字符串："<< name << "\t" << age << "\t" << birthday << endl; 
// 	// }
// 	return EXIT_SUCCESS;
// }

// // int main()
// // {
// // 	// 读文件
// // 	ifstream inFile("train.csv", ios::in);
// // 	string lineStr;
// // 	vector<vector<string> > strArray;
// // 	while (getline(inFile, lineStr))
// // 	{
// // 		// 打印整行字符串
// // 		cout << lineStr << endl;
// // 		// 存成二维表结构
// // 		stringstream ss(lineStr);
// // 		string str;
// // 		vector<string > lineArray;
// // 		// 按照逗号分隔
// // 		while (getline(ss, str, ','))
// // 			lineArray.push_back(str);
// // 		strArray.push_back(lineArray);
// //         // cout << strArray << endl;
// // 	}

// // 	// cout << strArray << endl;
// // 	// getchar();
// // 	return 0;
// // }

// 		// istringstream sin(line); //将整行字符串line读入到字符串流istringstream中
// 		// vector<string> fields; //声明一个字符串向量
// 		// string field;
// 		// while (getline(sin, field, ',')) //将字符串流sin中的字符读入到field字符串中，以逗号为分隔符
// 		// {
// 		// 	fields.push_back(field); //将刚刚读取的字符串添加到向量fields中