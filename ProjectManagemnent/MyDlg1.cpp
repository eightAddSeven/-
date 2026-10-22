// MyDlg1.cpp : implementation file
//

#include "stdafx.h"
#include "ProjectManagemnent.h"
#include "MyDlg1.h"


// CMyDlg1 dialog

IMPLEMENT_DYNAMIC(CMyDlg1, CDialog)

CMyDlg1::CMyDlg1(CWnd* pParent /*=NULL*/)
	: CDialog(CMyDlg1::IDD, pParent)
	, m_first(0)
	, m_second(0)
	, m_result(0)
{

}

CMyDlg1::~CMyDlg1()
{
}

void CMyDlg1::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_EDIT1, m_first);
	DDX_Text(pDX, IDC_EDIT2, m_second);
	DDX_Text(pDX, IDC_EDIT3, m_result);
}


BEGIN_MESSAGE_MAP(CMyDlg1, CDialog)
	ON_BN_CLICKED(IDOK, &CMyDlg1::OnBnClickedOk)
	ON_BN_CLICKED(IDC_BUTTON1, &CMyDlg1::OnBnClickedButton1)
END_MESSAGE_MAP()


// CMyDlg1 message handlers

void CMyDlg1::OnBnClickedOk()
{
	// TODO: Add your control notification handler code here
	AfxMessageBox("Hello");
	OnOK();
}

void CMyDlg1::OnBnClickedButton1()
{
	// TODO: Add your control notification handler code here
	UpdateData(TRUE);
	CString str1, str2;
	str1.Format("%f", m_first);
	str2.Format("%f", m_second);
	if ((str1.IsEmpty()) || (str2.IsEmpty()))
	{
		MessageBox("请输入有效数字");
		return;
	}
	m_result = m_first + m_second;
	UpdateData(FALSE);
}
