#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<cmath>
using namespace std;

vector<vector<double>>data;
void assign(int i,int t2,int t1,double& x,double&y,double&z)
{
    x=data[i][3*t2+1]-data[i][3*t1+1];
    y=data[i][3*t2+2]-data[i][3*t1+2];
    z=data[i][3*t2+3]-data[i][3*t1+3];
}
double ip(vector<double>v1,vector<double>v2)
{
    double a=v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2];
    double b=sqrt(v1[0]*v1[0]+v1[1]*v1[1]+v1[2]*v1[2]);
    double c=sqrt(v2[0]*v2[0]+v2[1]*v2[1]+v2[2]*v2[2]);
    return acos(a/(b*c));
}


int main()
{
    while(1)
    {
        data.clear();
        string name;
        cout<<"please input a xx for scuba_xx or quit(q)"<<endl;
        cin>>name;
        if(name=="q")break;
        
        ifstream fin;
        name="scuba_"+name+".txt";
        fin.open(name.c_str());
        if(!fin.is_open())cout<<"fuck!"<<endl;
        
        ofstream fout;
        name+="_f.txt";
        fout.open(name);
        
        vector<double>data2;
        double time,x,y,z;
        int count=0;
        while(1)
        {
            data2.clear();
            if(!(fin>>time)) break;
            data2.push_back(time);
            for(int i=0;i<15;i++)
            {
                fin>>x;
                fin>>y;
                fin>>z;
                data2.push_back(x);
                data2.push_back(y);
                data2.push_back(z);
            }
            fin>>x;
            fin>>y;
            fin>>z;
            data2.push_back(x);
            data2.push_back(y);
            data2.push_back(z);
            data.push_back(data2);
            count++;
        }
        //cout<< count<<endl;   
 
        vector<double> v1;
        vector<double> v2;
        vector<double> angle;
        v1.resize(3);
        v2.resize(3);
        double cangle;
        vector<double> velocity;
        for(size_t i=0;i<data.size();++i)
        {  
            angle.clear();
            assign(i,0,1,v1[0],v1[1],v1[2]);
            assign(i,2,1,v2[0],v2[1],v2[2]);
            cangle=ip(v1,v2);
            angle.push_back(cangle);
            assign(i,0,1,v1[0],v1[1],v1[2]);
            assign(i,3,1,v2[0],v2[1],v2[2]);
            cangle=ip(v1,v2);
            angle.push_back(cangle);
            assign(i,8,1,v1[0],v1[1],v1[2]);
            assign(i,2,1,v2[0],v2[1],v2[2]);
            cangle=ip(v1,v2);
            angle.push_back(cangle);
            assign(i,8,1,v1[0],v1[1],v1[2]);
            assign(i,3,1,v2[0],v2[1],v2[2]);
            cangle=ip(v1,v2);
            angle.push_back(cangle);
            assign(i,4,2,v1[0],v1[1],v1[2]);
            assign(i,1,2,v2[0],v2[1],v2[2]);
            cangle=ip(v1,v2);
            angle.push_back(cangle);
            assign(i,6,4,v1[0],v1[1],v1[2]);
            assign(i,2,4,v2[0],v2[1],v2[2]);
            cangle=ip(v1,v2);
            angle.push_back(cangle);
            assign(i,1,3,v1[0],v1[1],v1[2]);
            assign(i,5,3,v2[0],v2[1],v2[2]);
            cangle=ip(v1,v2);
            angle.push_back(cangle);
            assign(i,3,5,v1[0],v1[1],v1[2]);
            assign(i,7,5,v2[0],v2[1],v2[2]);
            cangle=ip(v1,v2);
            angle.push_back(cangle);
            assign(i,1,8,v1[0],v1[1],v1[2]);
            assign(i,9,8,v2[0],v2[1],v2[2]);
            cangle=ip(v1,v2);
            angle.push_back(cangle);
            assign(i,1,8,v1[0],v1[1],v1[2]);
            assign(i,10,8,v2[0],v2[1],v2[2]);
            cangle=ip(v1,v2);
            angle.push_back(cangle);
            assign(i,9,8,v1[0],v1[1],v1[2]);
            assign(i,10,8,v2[0],v2[1],v2[2]);
            cangle=ip(v1,v2);
            angle.push_back(cangle);
            assign(i,11,9,v1[0],v1[1],v1[2]);
            assign(i,8,9,v2[0],v2[1],v2[2]);
            cangle=ip(v1,v2);
            angle.push_back(cangle);
            assign(i,13,11,v1[0],v1[1],v1[2]);
            assign(i,9,11,v2[0],v2[1],v2[2]);
            cangle=ip(v1,v2);
            angle.push_back(cangle);
            assign(i,8,10,v1[0],v1[1],v1[2]);
            assign(i,12,10,v2[0],v2[1],v2[2]);
            cangle=ip(v1,v2);
            angle.push_back(cangle);
            assign(i,10,12,v1[0],v1[1],v1[2]);
            assign(i,14,12,v2[0],v2[1],v2[2]);
            cangle=ip(v1,v2);
            angle.push_back(cangle);
 
            for(size_t j=0;j<data[i].size();++j)fout<<data[i][j]<<" ";
            for(size_t j=0;j<angle.size();++j)fout<<angle[j]<<" ";
            
            velocity.clear();
            if(i==0)for(size_t j=0;j<data[i].size();++j)velocity.push_back(0);
            else
            {
                for(size_t j=0;j<data[i].size();++j)
                {
                    velocity.push_back((data[i][j]-data[i-1][j])/(data[i][0]-data[i-1][0]));
                }
            }
            for(size_t j=0;j<velocity.size();++j)fout<<velocity[j]<<" ";
            fout<<endl;
        }
        cout<<"there are "<<data.size()<<" time frame in "<<name<<endl<<endl;
    }
}


