#pragma once


// CDragAndDropListBox

class CDragAndDropListBox : public CListBox
{
	DECLARE_DYNAMIC(CDragAndDropListBox)

public:
	CDragAndDropListBox();
	virtual ~CDragAndDropListBox();

protected:
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnLButtonDown(UINT nFlags, CPoint point);
	afx_msg void OnLButtonUp(UINT nFlags, CPoint point);
	afx_msg void OnMouseMove(UINT nFlags, CPoint point);
	afx_msg void OnTimer(UINT_PTR nIDEvent);

private:
	int   m_MovingIndex;
	int   m_MoveToIndex;
	BOOL  m_Captured;
	DWORD m_Interval;

private:
	void InsertDragedItem(void);
	void DoTheScrolling(CPoint Point, CRect ClientRect);
	void DrawTheLines(int Index);
};


