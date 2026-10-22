#pragma once
#include "draganddroplistbox.h"


// CMyDlg dialog

class CMyDlg : public CDialog
{
	DECLARE_DYNAMIC(CMyDlg)

public:
	CMyDlg(CWnd* pParent = NULL);   // standard constructor
	virtual ~CMyDlg();

// Dialog Data
	enum { IDD = IDD_DIALOG1 };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
	afx_msg void OnLbnSelchangeList1();
	DECLARE_MESSAGE_MAP()
public:
	CDragAndDropListBox m_ListBox;
	virtual BOOL OnInitDialog();
};
