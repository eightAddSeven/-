#ifndef LINETEST_H
#define LINETEST_H

#pragma once
#include "PluginInterface.h"

const int nbFunc = 7;
const int TOOLBARCOUNT = 7;

class LineTest
{
public:
	LineTest(void);
	~LineTest(void);
	LineTest(char *pname, int num);
public:
	static void LineDesign(void);
	static void LineDesign_GIS(void);
	static void LineManagement(void);
	static void CannonRoadDesign(void);
	static void RemoveLines(void);
	static void LineEdit(void);
	static void HypickLine(void);


	char *GetPluginName(void);
	int GetMenuItemCount(void);
	MenuItem *GetMenuItem(void);
	static MenuItem menuItems[nbFunc];

	int GetToolBarCount(void);
	ToolBarItem *GetToolBarItem(void);
	static ToolBarItem arrToolBarItem[TOOLBARCOUNT];

	std::vector<HICON> GetPlugInIcon(void);

private:
	char *m_pName;
	int m_MenuItemCount;
	std::vector<HICON> m_ToolBarIcon;
};

#endif