#ifndef CPROJECT_H
#define CPROJECT_H

#pragma once
#include "PluginManager.h"
#include <vector>

const int MENUITEMS = 8;
const int TOOLBARCOUNT = 8;
class Project
{
public:
	Project(void);
	~Project(void);
	Project(char *pName, int num);
public:
	//菜单项对应的响应函数
	static void FuncNewProject(void);
	static void FuncOpenProject(void);
	static void FuncSaveProject(void);
	static void FuncCloseProject(void);
	static void FuncSetModule(void);
	static void FuncTranslate(void);
	static void FuncGetProjectInfo(void);
	static void FuncExit(void);
//

	char *GetPluginName(void);
	int GetMenuItemCount(void);
	MenuItem *GetMenuItem(void);
	static MenuItem arrMenuItems[MENUITEMS];

	int GetToolBarCount(void);
	ToolBarItem *GetToolBarItem(void);
	static ToolBarItem arrToolBarItem[TOOLBARCOUNT];

	std::vector<HICON> GetPlugInIcon(void);
	
private:
	char *m_pName;
	int  m_MenuCount;
	int  m_ToolBarCount;
};

#endif