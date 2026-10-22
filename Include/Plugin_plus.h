

#ifndef Plugin_plus_H
#define Plugin_plus_H


#include "Parameters.h"
#include "PluginManager.h"
#include <string>
using namespace std;


const std::string pluginName = "OBSPlugin";

//#define WM_LOADFILEBYPATH WM_USER

//const bool MODE_TRANSFER = true;
//const bool MODE_CLONE = false;

class Plugin_plus 
{
public:
	Plugin_plus(){};
	void init(PluginManager *pp, HMENU &hMenu, HWND &hWnd, CToolBar *pwndToolBar);
	virtual ~Plugin_plus()
	{
		(PluginParameters::GetInstance())->DestroyInstance();
	};
private:
	PluginManager _pluginsManager;
};

#endif 
