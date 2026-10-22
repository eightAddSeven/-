#ifndef PLUGININTERFACE_H
#define PLUGININTERFACE_H
#include <vector>
using namespace std;

#define  EXPORT __declspec(dllexport)
#define  IMPORT __declspec(dllimport)

const int MAX_LENGTH = 64;

struct ShortCutKey //快捷键结构体
{
	bool isCtrl;      //ctrl键
	bool isAlt;       //alt键
	bool isShift;     //shift键
	unsigned char key;//其他键
};

typedef void (*PMENUCMD)(void);       //菜单响应消息    
struct MenuItem //菜单项结构体
{
	char menuName[MAX_LENGTH];//菜单项名字
	PMENUCMD MenuCmd;         //菜单项对应的消息响应函数
	int nID;                  //菜单项的ID
	bool isCheck;             //是否对菜单项做标记
	ShortCutKey *pSHKey;      //菜单项对应的快捷键
};

struct PluginData
{
	HWND pluginHandel;
};

typedef void (*PTOOLBARCMD)(void);  //工具条响应函数

struct ToolBarItem//工具栏结构体
{
	char toolbarName[MAX_LENGTH];
	PTOOLBARCMD ToolBarCmd;
	int nID;
};



typedef const char *(*PFUNCGETNAME)(void);            //获取插件的名字
      
typedef MenuItem *(*PFUNCGETFUNCSARRAY)(int *);       //获取菜单项
typedef ToolBarItem *(*PFUNCGetTOOLBARARRAY)(int *);  //获取工具条的内容
            
typedef UINT (*PFUNCINITPLUGIN)(CToolBar *pwndToolBar);
typedef vector<HICON> &(*PFUNCGETPLUGINRESOURCE)(void);

//extern "C" EXPORT const char *GetName(void);
//extern "C" EXPORT MenuItem *GetFuncsArray(int *);

#endif