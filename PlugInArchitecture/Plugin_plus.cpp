#include "stdafx.h"
#include "Plugin_plus.h"
#include "PluginInterface.h"
#pragma comment(lib,"ws2_32.lib")

void Plugin_plus::init(PluginManager *pp, HMENU &hMenu, HWND &hWnd, CToolBar *pwndToolBar)
{
	PluginParameters *pluginParam = PluginParameters::GetInstance();
	PluginData pluginData;
	pluginData.pluginHandel = hWnd;

	char *p = "";
	pp->Init(pluginData);
	/*_pluginsManager.loadPlugins(p);
	_pluginsManager.setMenu(hMenu, pluginName.c_str());*/
	pp->LoadPlugins(p);
	pp->SetMenu(hMenu, pluginName.c_str());
//	pp->GetToolBarInfo();
//	pp->ReDisplayToolbar(pwndToolBar);
	//_pluginsManager.SetMenu(AfxGetMainWnd()->m_hWnd, pluginName.c_str());

}