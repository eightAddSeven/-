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
	ToolBarCommand(const char *toolbarName, int id, PTOOLBARCMD pCmd):
		nID(id)
		,pToolBarFunc(pCmd)
	{
		strcpy(toolBarName, toolbarName);
	}
};

struct PluginInfo 
{
	HINSTANCE hInstance;
	HMENU pluginMenu;
	HICON pluginHicon;

	PFUNCGETNAME           GetName;
	PFUNCGETFUNCSARRAY     GetMenuArray;
	PFUNCGetTOOLBARARRAY   GetToolBarArray;
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
	void SetMenu(HMENU hMenu, const char *menuName);
	void SetButton(CToolBar *pwndToolBar);
	void GetCustomizedShortCut(char *pluginName, MenuItem *pItem, int nMenuItemsCount);
	void ReDisplayToolbar(CToolBar *pwndToolBar);
	void GetToolBarInfo(void);
	void SetToolBar(CToolBar *pwndToolBar);
	map<HICON, char *> GetIconInfo(void);
	std::vector<HICON>          m_pluginhIcon;
private:
	bool HasPlugin(void) const;
private:
	int        m_pluginCount;
	PluginData m_data;
	HMENU      m_pluginMenu;
	std::vector<PluginInfo *>   m_pluginInfo;
	std::vector<PluginCommand>  m_pluginCommands;
	std::vector<ToolBarCommand> m_toolBarCommands;
	
	std::map<HICON, char *>   m_pluginMap;
public:
	CToolBar *m_pToolBar;
protected:

};


#endif