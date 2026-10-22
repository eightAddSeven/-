#include "StdAfx.h"
#include "Project.h"
#include "MyDlg.h"
#include "MyDlg1.h"


Project::Project(void)
{
}

Project::~Project(void)
{
}

Project::Project(char *pName, int num)
{
	m_pName = pName;
	m_MenuCount = num;
}

MenuItem Project::arrMenuItems[MENUITEMS] = {
	{ _T("新建工程"),      FuncNewProject,      0, false, NULL},
	{ _T("打开工程"),      FuncOpenProject,     0, false, NULL},
	{ _T("保存工程"),      FuncSaveProject,     0, false, NULL},
	{ _T("关闭工程"),      FuncCloseProject,    0, false, NULL},
	{ _T("模块设置"),      FuncSetModule,       0, false, NULL},
	{ _T("投影及坐标变换"),FuncTranslate,       0, false, NULL},
	{ _T("工程信息"),      FuncGetProjectInfo,  0, false, NULL},
	{ _T("退出"),          FuncExit,            0, false, NULL}
};


ToolBarItem Project::arrToolBarItem[TOOLBARCOUNT] = {
	{ _T("新建工程"),      FuncNewProject,      0},
	{ _T("打开工程"),      FuncOpenProject,     0},
	{ _T("保存工程"),      FuncSaveProject,     0},
	{ _T("关闭工程"),      FuncCloseProject,    0},
	{ _T("模块设置"),      FuncSetModule,       0},
	{ _T("投影及坐标变换"),FuncTranslate,       0},
	{ _T("工程信息"),      FuncGetProjectInfo,  0},
	{ _T("退出"),          FuncExit,            0}
};

void Project::FuncNewProject()
{
	//AfxMessageBox("点击了新建工程选项");
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	//CMyDlg dlg;
	//dlg.DoModal();

	/*if (IDOK == dlg.DoModal())
	{
		AfxMessageBox("显示对话框成功");
		return;
	}*/
	//dlg.DoModal();
}

void Project::FuncOpenProject()
{
	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	CMyDlg1 dlg;
	dlg.DoModal();
	AfxMessageBox("点击了打开工程选项");
}

void Project::FuncSaveProject()
{
	AfxMessageBox("点击了保存工程选项");
}

void Project::FuncCloseProject()
{
	AfxMessageBox("点击了关闭工程选项");
}

void Project::FuncGetProjectInfo()
{
	AfxMessageBox("点击了获取工程信息选项");

}

void Project::FuncSetModule()
{
	AfxMessageBox("点击了设置模块信息选项");
}

void Project::FuncTranslate()
{
	AfxMessageBox("点击了投影变换选项");
}

void Project::FuncExit()
{
	AfxMessageBox("点击了退出选项");
}

char *Project::GetPluginName(void)
{
	return m_pName;
}

int Project::GetMenuItemCount(void)
{
	return m_MenuCount;
}

MenuItem *Project::GetMenuItem(void)
{
	return arrMenuItems;
}

int Project::GetToolBarCount(void)
{
	return m_ToolBarCount;
}

ToolBarItem *Project::GetToolBarItem(void)
{
	return arrToolBarItem;
}


//vector<HICON> Project::GetPlugInIcon()
//{
//	return m_ToolBarIcon;
//}