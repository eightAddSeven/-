#include "StdAfx.h"
#include "LineTest.h"

LineTest::LineTest(void)
{
}

LineTest::~LineTest(void)
{
}

MenuItem LineTest::menuItems[nbFunc] = {
	{_T("测线设计"),     LineDesign,       0, false, NULL},
	{_T("测线设计GIS"),  LineDesign_GIS,   0, false, NULL},
	{_T("测线管理"),     LineManagement,   0, false, NULL},
	{_T("测线炮点设计"), CannonRoadDesign, 0, false, NULL},
	{_T("删除所有测线"), RemoveLines,      0, false, NULL},
	{_T("测线编辑"),     LineEdit,         0, false, NULL},
	{_T("HYPICK测线"),   HypickLine,       0, false, NULL}
};


ToolBarItem LineTest::arrToolBarItem[TOOLBARCOUNT] = {
	{_T("测线设计"),     LineDesign,       0},
	{_T("测线设计GIS"),  LineDesign_GIS,   0},
	{_T("测线管理"),     LineManagement,   0},
	{_T("测线炮点设计"), CannonRoadDesign, 0},
	{_T("删除所有测线"), RemoveLines,      0},
	{_T("测线编辑"),     LineEdit,         0},
	{_T("HYPICK测线"),   HypickLine,       0}
};

LineTest::LineTest(char *pname, int num)
{
	m_pName = pname;
	m_MenuItemCount = num;
}

void LineTest::LineDesign(void)
{
	AfxMessageBox("测线设计");
}

void LineTest::LineDesign_GIS(void)
{
	AfxMessageBox("测线设计GIS");
}

void LineTest::LineManagement(void)
{
	AfxMessageBox("测线管理");
}

void LineTest::CannonRoadDesign(void)
{
	AfxMessageBox("测线炮点设计");
}

void LineTest::RemoveLines(void)
{
	AfxMessageBox("删除所有测线");
}

void LineTest::LineEdit(void)
{
	AfxMessageBox("测线编辑");
}

void LineTest::HypickLine(void)
{
	AfxMessageBox("HYPICK测线");
}

char *LineTest::GetPluginName(void)
{
	return m_pName;
}

int LineTest::GetMenuItemCount(void)
{
	return m_MenuItemCount;
}

MenuItem *LineTest::GetMenuItem()
{
	return menuItems;
}

int LineTest::GetToolBarCount(void)
{
	return TOOLBARCOUNT;
}

ToolBarItem *LineTest::GetToolBarItem(void)
{
	return arrToolBarItem;
}
