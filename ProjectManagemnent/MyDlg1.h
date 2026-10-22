#pragma once


// CMyDlg1 dialog

class CMyDlg1 : public CDialog
{
	DECLARE_DYNAMIC(CMyDlg1)

public:
	CMyDlg1(CWnd* pParent = NULL);   // standard constructor
	virtual ~CMyDlg1();

// Dialog Data
	enum { IDD = IDD_DIALOG2 };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedOk();
	float m_first;
	float m_second;
	float m_result;
	afx_msg void OnBnClickedButton1();
};
