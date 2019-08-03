/*
此代码用于将排序后的刘泽奇的降维结果进行两两用户之间的相似度计算 
*/
#include<cstdio>  
#include<cstring>  
#include<algorithm>  
#include<iostream>  
#include<string>  
#include<vector>  
#include<stack>  
#include<bitset>  
#include<cstdlib>  
#include<cmath>  
#include<set>  
#include<list>  
#include<deque>  
#include<map>  
#include<queue>  
#include<fstream>
using namespace std; 

typedef long long ll;
const double PI = acos(-1.0);
const double eps = 1e-6;

struct node
{
	long name;
	double data[4];
	node *next;
};

node *head=NULL;

bool reading_data()
{
	head=new node();
	head->name=0;//头结点的name用于节点总数计数
	 
	fstream f1;
    f1.open("aa_at_pre4_sort.txt", ios::in);
    
    node *p=head;
    
    while(!f1.eof())
    {
    	long tname;
    	double tdata[4];
    	f1>>tname>>tdata[0]>>tdata[1]>>tdata[2]>>tdata[3];
    	
    	node *t=new node();
    	t->name=tname;
    	t->data[0]=tdata[0];
    	t->data[1]=tdata[1];
    	t->data[2]=tdata[2];
    	t->data[3]=tdata[3];
    	
    	p->next=t;
    	p=p->next;
    	(head->name)++;
	}
	
	return true;
}

bool calculating()
{
    fstream fout("embedcos.txt", ios::out);
    
    node *p=head->next;
    
    while(p)
    {
    	//cout<<"calculating : "<<p->name<<endl;
    	
    	node *q=head->next;
    	while(q)
    	{
    		double sum=0;
    		double a2=0;
    		double b2=0;
    	
    		for(int i=0;i<4;i++)
    		{
    			sum+=(p->data[i])*(q->data[i]);
    			a2+=(p->data[i])*(p->data[i]);
    			b2+=(q->data[i])*(q->data[i]);
			}
			
			double cos=sum/( sqrt(a2) * sqrt(b2) );
			
			if(cos>0.9&&abs(cos-1)>eps)
				fout<<p->name<<" "<<q->name<<" "<<cos<<endl;
			
			q=q->next;
		}
    	
    	p=p->next;
    	/*
    	if(p->name>4000)
			return true;
		*/
	}
    
    
    return true;
}

int main()
{
	cout<<"reading..."<<endl;
	reading_data();
	cout<<"have read:"<<head->name<<endl;
	
	calculating();
	
	return 0;
}
