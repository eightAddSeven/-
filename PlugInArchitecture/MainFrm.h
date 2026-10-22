// MainFrm.h : interface of the CMainFrame class
//


#pragma once
#include "PluginManager.h"
#include "Plugin_plus.h"

class CMainFrame : public CMDIFrameWnd
{
	DECLARE_DYNAMIC(CMainFrame)
public:
	CMainFrame();

// Attributes
public:

// Operations
public:

// Overrides
public:
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);

// Implementation
public:
	virtual ~CMainFrame();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:  // control bar embedded members
	CStatusBar  m_wndStatusBar;
	CToolBar    m_wndToolBar;

	CToolBar    m_pluginToolbar;
	PluginManager *m_pluginManager;
	HMENU m_hMenu;

private:
	CImageList m_imageList;
	list<vector<HICON>> m_pluginhIcon;
	vector<HICON> m_tmphIcon;

public:
	void AddButtonToToolBar();
// Generated message map functions
protected:
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg void OnMenuExcute(UINT nID);
	afx_msg void OnToolBarExcute(UINT nID);
	//afx_msg void  OnToolBT1();
	//afx_msg void  OnToolBT2();
	//afx_msg void  OnToolBT3();
	//afx_msg void  OnToolBT4();
	//afx_msg void  OnToolBT5();
	DECLARE_MESSAGE_MAP()
	virtual BOOL OnCommand(WPARAM wParam, LPARAM lParam);
};


