#include "StdAfx.h"
#include "ViewTest.h"

CViewTest::CViewTest(void)
{
}

CViewTest::~CViewTest(void)
{
}

MenuItem CViewTest::funcItem[nbFunc] = {
	{ _T("二位监控显示"),   Display2d,       0, false, NULL },
	{ _T("三维监控显示"),   Display3d,       0, false, NULL },
	{ _T("菜单栏"),         DisplayMenuItem, 0, false, NULL },
	{ _T("标准工具栏"),     DisplayToolBar,  0, false, NULL },
	{ _T("状态栏"),         DisplayStateBar, 0, false, NULL },
	{ _T("文本框"),         DisplayText,     0, false, NULL }
};

ToolBarItem CViewTest::arrToolBarItem[TOOLBARCOUNT] = {
	{ _T("二位监控显示"),   Display2d,       0},
	{ _T("三维监控显示"),   Display3d,       0},
	{ _T("菜单栏"),         DisplayMenuItem, 0},
	{ _T("标准工具栏"),     DisplayToolBar,  0},
	{ _T("状态栏"),         DisplayStateBar, 0},
	{ _T("文本框"),         DisplayText,     0}
};

CViewTest::CViewTest(char *pt, int num)
{
	pName = pt;
	nbFuncItem = num;
}

void CViewTest::InitMenuItem()
{
}

void CViewTest::Display2d()
{
	AfxMessageBox("二位监控显示");
}

void CViewTest::Display3d()
{
	AfxMessageBox("三位监控显示");
}

void CViewTest::DisplayMenuItem()
{
	AfxMessageBox("菜单栏");
}

void CViewTest::DisplayToolBar()
{
	AfxMessageBox("二标准工具栏");
}

void CViewTest::DisplayStateBar() 
{
	AfxMessageBox("状态栏");
}

void CViewTest::DisplayText()
{
	AfxMessageBox("文本框");
}

char *CViewTest::GetPluginName()
{
	return pName;
}

int CViewTest::GetMenuItemCount()
{
	return nbFuncItem;
}

MenuItem *CViewTest::GetMenuItem()
{
	return funcItem;
}

ToolBarItem * CViewTest::GetToolBarItem()
{
	return arrToolBarItem;
}

int CViewTest::GetToolBarCount()
{
	return TOOLBARCOUNT;
}