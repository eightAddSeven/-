#include "stdafx.h"
#include "PluginManager.h"


#define MENU_POSITION_ERR (UINT)-1 

#define USERMSG "This plugin is not compatible with current version of PluginTest.\n\
Remove this plugin from plugins directory if you don't want to see this message on the next launch time."


PluginManager::PluginManager():m_pluginMenu(0)
{
	pluginParam = PluginParameters::GetInstance();
}

PluginManager::~PluginManager()
{
	for (int i=0; i<m_pluginInfo.size(); i++)
	{
		delete m_pluginInfo[i];
	}

	if (m_pluginMenu != NULL)
	{
		DestroyMenu(m_pluginMenu);
	}
}

void PluginManager::Init(PluginData &data)
{
	m_data = data;
	pluginParam->LoadMenuItemFromXML(m_MenuItems);
	pluginParam->LoadPlugInsFromXML(m_PlugInItems);
}

bool PluginManager::LoadPlugins(const char *pPath)
{
	vector<HICON> tmphIconVec;

	/*char pluginPath[PATH_LENGTH];
	::GetModuleFileName(NULL, pluginPath, sizeof(pluginPath));
	PathRemoveFileSpec(pluginPath);
	std::vector<string> dllNames;
	string pluginFullPathFilter = (pPath && pPath[0])?pPath:pluginPath;
	pluginFullPathFilter += "\\plugins\\*.dll";
	WIN32_FIND_DATA foundData;
	HANDLE hFindFile = ::FindFirstFile(pluginFullPathFilter.c_str(), &foundData);
	
	if (hFindFile != INVALID_HANDLE_VALUE)
	{
		string pluginFullPath = (pPath && pPath[0])?pPath:pluginPath;
		pluginFullPath += "\\plugins\\";
		pluginFullPath += foundData.cFileName;
		dllNames.push_back(pluginFullPath);

		while (::FindNextFile(hFindFile, &foundData))
		{
			string fullPath = (pPath && pPath[0])?pPath:pluginPath;
			fullPath += "\\plugins\\";
			fullPath += foundData.cFileName;
			dllNames.push_back(fullPath);
		}
	}

	::FindClose(hFindFile);*/
	int i=0;
	//HICON hIcon;
	//char *szhIconStr;

	/*for (int i=0; i<m_PlugInItems.size(); i++)
	{

	}*/
	for (; i</*dllNames.size()*/m_PlugInItems.size(); i++)
	{
		//char *pt = m_PlugInItems.at(i).pPath + m_PlugInItems.at(i).pName;
		strcat(m_PlugInItems.at(i).pPath, "\\");
		strcat(m_PlugInItems.at(i).pPath,  m_PlugInItems.at(i).pName);
		char *pt = m_PlugInItems.at(i).pPath;
		PluginInfo *p = new PluginInfo();
		try
		{
			char tempStr[MAX_PATH];
			strcpy(tempStr, /*dllNames[i].c_str()*/pt);
			strcpy(p->pluginName, PathFindFileName(tempStr));
			p->hInstance = ::LoadLibrary(/*dllNames[i].c_str()*/pt);
			if (p->hInstance == NULL)
			{
				throw string("Load Library is failed.\nMake \"Runtime Library\" setting of this project as \"Multi-threaded(/MT)\" may cure this problem.");
			}

			p->GetName = (PFUNCGETNAME)GetProcAddress(p->hInstance, "GetName");
			if (NULL == p->GetName)
			{
				throw string("Missing \"setInfo\" function");
			}

			
			p->GetMenuArray  = (PFUNCGETFUNCSARRAY)GetProcAddress(p->hInstance, "GetMenuArray");
			if (NULL == p->GetMenuArray)
			{
				throw string("Missing\"getFuncsArray\" function");
			}

			p->GetToolBarArray = (PFUNCGETTOOLBARARRAY)GetProcAddress(p->hInstance, "GetToolBarArray");
			if (NULL == p->GetToolBarArray)
			{
				throw string("Missing\"getToolBarFuncArray\" function");
			}

			p->toolbarItems = p->GetToolBarArray(&p->nToolBarCount);


			p->menuItems = p->GetMenuArray(&p->nMenuItemCount);
			if (p->menuItems == NULL || p->nMenuItemCount <= 0)
			{
				throw string("Missing \"FuncItems\" array, or the nb of Function Item is not set correctly");

			}

			p->GetPluginIcon = (PFUNCGETPLUGINRESOURCE)GetProcAddress(p->hInstance, "GetPlugInIcon");
			if (NULL == p->GetPluginIcon)
			{
				throw string("Missing\"getPluginResource\" function");
			}

			 p->GetPluginIcon(tmphIconVec);
			 m_pluginhIcon.push_back(tmphIconVec);

			 for (int i=0; i<tmphIconVec.size(); i++)
			 {
				 m_tmphIcon.push_back(tmphIconVec.at(i));
			 }

			 //tmphIconVec.clear();

			//m_pluginMap.insert(pair<HICON, char *>(hIcon, szhIconStr));
			for (int i = 0 ; i<p->nMenuItemCount ; i++)
			{
				//PFUNCPLUGINCMD */temp = pi->_funcItems[i].pFunc;
				if (!p->menuItems[i].MenuCmd)
				{
					throw string("\"FuncItems\" array is not set correctly");
				}
			}

			for (int j=0; j<p->nToolBarCount; j++)
			{
				if (!p->toolbarItems[j].ToolBarCmd)
				{
					throw string("\"ToolBarItems\" array is not set correctly");
				}
			}

			p->pluginMenu = ::CreateMenu();
			m_pluginInfo.push_back(p);

		}
		catch(string s)
		{
			s += "\n\n";
			s += USERMSG;
			::MessageBox(NULL, s.c_str(), pt/*dllNames[i].c_str()*/, MB_OK);
			delete p;
		}
		catch(...)
		{
			string msg = "Fail loaded";
			msg += "\n\n";
			msg += USERMSG;
			::MessageBox(NULL, msg.c_str(), pt/*dllNames[i].c_str()*/, MB_OK);
			delete p;
		}
		
	}

	return true;
}

void PluginManager::RunPluginCommand(int i)
{
	if (i<m_pluginCommands.size())
	{
		m_pluginCommands[i].pFunc();
	}
}

void PluginManager::RunPluginCommand(const char *pluginName, int commandID)
{
	for (int i=0; i<m_pluginCommands.size(); i++)
	{
		if (!stricmp(m_pluginCommands[i].pluginName, pluginName))
		{
			if (m_pluginCommands[i].nID == commandID)
			{
				m_pluginCommands[i].pFunc();
			}
		}
	}
}

void PluginManager::RunToolBarCommand(int commandID)
{
	if (commandID<m_toolBarCommands.size())
	{
		m_toolBarCommands[commandID].pToolBarFunc();
	}
}

void PluginManager::RunToolBarCommand(const char *pToolBarName, int commandID)
{
	for (int i=0; i<m_toolBarCommands.size(); i++)
	{
		if (!stricmp(m_toolBarCommands[i].toolBarName, pToolBarName))
		{
			if (m_toolBarCommands[i].nID == commandID)
			{
				m_toolBarCommands[i].pToolBarFunc();
			}
		}
	}
}

void PluginManager::SetMenu(HMENU hMenu, const char *menuName)
{
	if (HasPlugin())
	{
		vector<PluginCmdShortcut> & pluginCmdSCList = (PluginParameters::GetInstance())->GetPluginCommandList();
	//	const char *nom_menu = (menuName && menuName[0])?menuName:"OBSPlugins";
		m_pluginMenu = ::CreateMenu();
		int n = ::GetMenuItemCount(hMenu);
		//const int n = 4;
		//::InsertMenu(hMenu, n+1, MF_POPUP | MF_BYPOSITION, (UINT)_hPluginsMenu, nom_menu);

		int m = m_pluginInfo.size();
		for (size_t i=0; i<m; i++)
		{
			//string name = m_pluginInfo[i]->
			//int temp = m_pluginInfo[i]->nMenuItemCount;
			::InsertMenu(hMenu, n+i, MF_BYPOSITION | MF_POPUP, (UINT_PTR)m_pluginInfo[i]->pluginMenu, m_MenuItems.at(i).c_str()/*m_pluginInfo[i]->GetName()*/);
			for (int j=0; j < m_pluginInfo[i]->nMenuItemCount; j++)
			{
				m_pluginCommands.push_back(PluginCommand(m_pluginInfo[i]->pluginName, j, m_pluginInfo[i]->menuItems[j].MenuCmd));
				int cmdID = ID_PLUGINS_CMD + (m_pluginCommands.size() - 1);
				m_pluginInfo[i]->menuItems[j].nID = cmdID;
				string itemName = m_pluginInfo[i]->menuItems[j].menuName;
				::InsertMenu(m_pluginInfo[i]->pluginMenu, j, MF_BYPOSITION, cmdID, itemName.c_str());
				/*if (!_pluginInfos[i]->_funcItems[j]._init2Check)
				::CheckMenuItem(_hPluginsMenu, cmdID, MF_BYCOMMAND | MF_CHECKED);*/

			}
		}

	}
}

void PluginManager::SetToolBarButtonID()
{
	if (HasPlugin())
	{
		for (int i=0; i<m_pluginInfo.size(); i++)
		{
			for (int j=0; j<m_pluginInfo[i]->nToolBarCount; j++)
			{
				m_toolBarCommands.push_back(ToolBarCommand(m_pluginInfo[i]->toolbarItems[j].toolbarName, j, m_pluginInfo[i]->toolbarItems[j].ToolBarCmd));
				int cmdID = ID_TOOLBAR_CMD + (m_toolBarCommands.size()-1);
				m_pluginInfo[i]->toolbarItems[j].nID = cmdID;
			}
		}
	}
}


void PluginManager::GetCustomizedShortCut(char *pluginName, MenuItem *pItem, int nMenuItemsCount)
{
	
}

bool PluginManager::HasPlugin(void) const
{
	return m_pluginInfo.size() != 0;
}

map<HICON, char *> PluginManager::GetIconInfo()
{
	return m_pluginMap;
}