// MyDlg.cpp : implementation file
//

#include "stdafx.h"
#include "ProjectManagemnent.h"
#include "MyDlg.h"


// CMyDlg dialog

IMPLEMENT_DYNAMIC(CMyDlg, CDialog)

CMyDlg::CMyDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CMyDlg::IDD, pParent)
{
	
}

CMyDlg::~CMyDlg()
{
}

void CMyDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
}


BEGIN_MESSAGE_MAP(CMyDlg, CDialog)
	 ON_LBN_SELCHANGE(IDC_LIST1, OnLbnSelchangeList1)
END_MESSAGE_MAP()


// CMyDlg message handlers

BOOL CMyDlg::OnInitDialog()
{
	CDialog::OnInitDialog();

	AFX_MANAGE_STATE(AfxGetStaticModuleState());
	for (int i=0; i<5; i++)
	{
		CString Text;
		Text.Format("ITem %d", i+1);
		try
		{
			m_ListBox.AddString(Text);
		}
		catch (CException* e)
		{
			TCHAR szCause[255];
			e->GetErrorMessage(szCause, 255);
			AfxMessageBox(szCause);
		}
		
	}
	// TODO:  Add extra initialization here

	return TRUE;  // return TRUE unless you set the focus to a control
	// EXCEPTION: OCX Property Pages should return FALSE
}


void CMyDlg::OnLbnSelchangeList1()
{
	CString str;
	int Index = m_ListBox.GetCurSel();
	m_ListBox.GetText(Index, str);
	//m_Selected.SetWindowText(str);
}