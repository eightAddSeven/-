#ifndef VIEWTEST_H
#define VIEWTEST_H

#pragma once
#include "PluginInterface.h"

#define V_2D_MONITER     0
#define V_3D_MONITER     1
#define V_MENUBAR        2
#define V_TOOLBAR        3
#define V_STATIONBAR     4
#define V_TEXT           5

//const char *pPluginName = " ”Õºπ‹¿Ì";
const int nbFunc = 6;
const int TOOLBARCOUNT = 6;

class CViewTest
{
public:
	CViewTest(void);
	~CViewTest(void);
	CViewTest(char *pt, int num);
public:
	static void Display2d(void);
	static void Display3d(void);
	static void DisplayMenuItem(void);
    static void DisplayToolBar(void) ;
    static void DisplayStateBar(void);
	static void DisplayText(void);

    char *GetPluginName(void);
	int GetMenuItemCount(void);
	static MenuItem *GetMenuItem(void);
	static MenuItem funcItem[nbFunc];

	int GetToolBarCount(void);
	ToolBarItem *GetToolBarItem(void);
	static ToolBarItem arrToolBarItem[TOOLBARCOUNT];

	std::vector<HICON> GetPlugInIcon(void);
private:
	char *pName;
	int nbFuncItem;
	std::vector<HICON> m_ToolBarIcon;
	
private:
	void InitMenuItem(void);

};

#endif
