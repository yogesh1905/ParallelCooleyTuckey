#include <bits/stdc++.h>
#include <math.h>
#include <fstream>
#include <iostream>
using namespace std;
using cd = complex<double>;
const double PI = acos(-1);
#define ll long long
void fft_serial(vector<cd> & a, bool invert,int n) {
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;

        if (i < j)
            swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * PI / len * (invert ? -1 : 1);
        cd wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            cd w(1);
            for (int j = 0; j < len / 2; j++) {
                cd u = a[i+j], v = a[i+j+len/2] * w;
                a[i+j] = u + v;
                a[i+j+len/2] = u - v;
                w *= wlen;
            }
        }
    }
    if (invert) {
        for (cd & x : a)
            x /= n;
    }
}

int main()
{
    std::vector<cd> x;
    int n;
    cin>>n;
    for (ll i = 0; i < pow(2,n); ++i)
    {
        cd temp(i,i+3);
        x.push_back(temp);
    }
    ofstream myfile;
    string fname="input.txt";
    int cn=1;
    myfile.open(fname, std::ios_base::app);
    for (int i = 1; i < pow(2,n); i=i*2)
    {  
        cout<<" n is "<<i<<endl;
        ll len=i;
        clock_t start, end; 
        start = clock(); 
        fft_serial(x,0,len);
        end = clock(); 
        double time_taken = double(end - start) / double(CLOCKS_PER_SEC); 
        myfile <<cn<<" "<<time_taken<<endl;
        cout<<"Time taken is "<<time_taken<<endl;
        cn+=1;
    }
    myfile.close();
}
