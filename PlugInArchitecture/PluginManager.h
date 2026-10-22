#ifndef PLUGINMANAGER_H
#define PLUGINMANAGER_H

#include "Resource.h"
#include "Parameters.h"
#include "PluginInterface.h"
#include "ShortCut.h"
#include <iostream>
#include <vector>
#include <Shlwapi.h>
#include <map>
#include <list>

using namespace std;

const int ID_COUNT = 4;
const int PATH_LENGTH = 256;

struct PluginCommand 
{
	char pluginName[LENGTH];
	int nID;
	PMENUCMD pFunc;
	PluginCommand(const char *pluginname, int id, PMENUCMD pCmd):
		nID(id),
		pFunc(pCmd)
		{
			strcpy(pluginName, pluginname);
		};
};

struct ToolBarCommand 
{
	char toolBarName[LENGTH];
	int nID;
	PTOOLBARCMD pToolBarFunc;
	ToolBarCommand(const char *toolbarName1, int id, PTOOLBARCMD pCmd):
		nID(id),
		pToolBarFunc(pCmd)
	{
		strcpy(toolBarName, toolbarName1);
	};
};

struct PluginInfo 
{
	HINSTANCE hInstance;
	HMENU pluginMenu;
	HICON pluginHicon;

	PFUNCGETNAME           GetName;
	PFUNCGETFUNCSARRAY     GetMenuArray;
	PFUNCGETTOOLBARARRAY   GetToolBarArray;
	PFUNCINITPLUGIN        InitPlugin;
	PFUNCGETPLUGINRESOURCE GetPluginIcon;

	MenuItem *menuItems;
	ToolBarItem *toolbarItems;
	int nMenuItemCount;
	int nToolBarCount;
	char pluginName[LENGTH];
	char toolbarName[LENGTH];

	PluginInfo():hInstance(NULL)
		,pluginMenu(NULL)
		,pluginHicon(NULL)
		,GetName(NULL)
		,menuItems(NULL)
		,toolbarItems(NULL)
		,nMenuItemCount(0)
	{
	}

	~PluginInfo()
	{
		if (pluginMenu != NULL)
		{
			::DestroyMenu(pluginMenu);
		}
		if (pluginHicon != NULL)
		{
			::DestroyIcon(pluginHicon);
		}
		if (hInstance != NULL)
		{
			::FreeLibrary(hInstance);
		}
	}
		
};

class PluginManager
{
public:
	PluginManager();
	~PluginManager();
	void Init(PluginData &data);
	bool LoadPlugins(const char *pPath);
	void RunPluginCommand(int i);
	void RunPluginCommand(const char *pluginName, int commandID);

	void RunToolBarCommand(int commandID);
	void RunToolBarCommand(const char *pToolBarName, int commandID);

	void SetMenu(HMENU hMenu, const char *menuName);
	void SetButton(CToolBar *pwndToolBar);
	void GetCustomizedShortCut(char *pluginName, MenuItem *pItem, int nMenuItemsCount);
	void ReDisplayToolbar(CToolBar *pwndToolBar);
	void SetToolBarButtonID(void);
	void SetToolBar(CToolBar *pwndToolBar);
	map<HICON, char *> GetIconInfo(void);
	list<vector<HICON>>       m_pluginhIcon;
	vector<HICON> GethIcon(void) {return m_tmphIcon; };
private:
	bool HasPlugin(void) const;
private:
	PluginParameters *pluginParam;
	int        m_pluginCount;
	PluginData m_data;
	HMENU      m_pluginMenu;
	std::vector<PluginInfo *>   m_pluginInfo;
	std::vector<PluginCommand>  m_pluginCommands;
	std::vector<ToolBarCommand> m_toolBarCommands;
	std::vector<HICON>          m_tmphIcon;
	std::map<HICON, char *>   m_pluginMap;
	vector<std::string> m_MenuItems;
	vector<PlugInConfigurationInfo> m_PlugInItems;
public:
	CToolBar *m_pToolBar;
protected:

};


#endif