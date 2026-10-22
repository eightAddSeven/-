// DragAndDropListBox.cpp : implementation file
//

#include "stdafx.h"
#include "ProjectManagemnent.h"
#include "DragAndDropListBox.h"

#define TID_SCROLLDOWN 100
#define TID_SCROLLUP   101
// CDragAndDropListBox

IMPLEMENT_DYNAMIC(CDragAndDropListBox, CListBox)

CDragAndDropListBox::CDragAndDropListBox()
:m_Captured(FALSE)
,m_Interval(0)
,m_MoveToIndex(-1)
,m_MovingIndex(-1)
{

}

CDragAndDropListBox::~CDragAndDropListBox()
{
}


BEGIN_MESSAGE_MAP(CDragAndDropListBox, CListBox)
	ON_WM_LBUTTONDOWN()
	ON_WM_LBUTTONUP()
	ON_WM_MOUSEMOVE()
	ON_WM_TIMER()
END_MESSAGE_MAP()



// CDragAndDropListBox message handlers



void CDragAndDropListBox::OnLButtonDown(UINT nFlags, CPoint point)
{
	// TODO: Add your message handler code here and/or call default

	CListBox::OnLButtonDown(nFlags, point);
	m_MoveToIndex = LB_ERR;
	m_MovingIndex = LB_ERR;
	m_Captured = FALSE;
	m_Interval = 0;
	BOOL Outside;
	int Index = ItemFromPoint(point, Outside);
	if (Index != LB_ERR && !Outside)
	{
		m_MovingIndex = Index;
		SetCurSel(Index);
	}
}

void CDragAndDropListBox::OnLButtonUp(UINT nFlags, CPoint point)
{
	// TODO: Add your message handler code here and/or call default

	if (m_MovingIndex != LB_ERR && m_Captured)
	{
		KillTimer(TID_SCROLLDOWN);
		KillTimer(TID_SCROLLUP);
		m_Captured = 0;
		m_Captured = FALSE;
		ReleaseCapture();

		CRect Rect;
		GetClientRect(&Rect);
		if (Rect.PtInRect(point))
		{
			InsertDragedItem();
		}
		else
		{
			Invalidate();
			UpdateWindow();
		}
		m_MoveToIndex = LB_ERR;
		m_MovingIndex = LB_ERR;
	}
	CListBox::OnLButtonUp(nFlags, point);
}

void CDragAndDropListBox::OnMouseMove(UINT nFlags, CPoint point)
{
	// TODO: Add your message handler code here and/or call default

	CListBox::OnMouseMove(nFlags, point);
	if (nFlags & MK_LBUTTON)
	{
		if (m_MovingIndex != LB_ERR && !m_Captured)
		{
			SetCapture();
			m_Captured = TRUE;
		}
		BOOL Outside;
		int Index = ItemFromPoint(point, Outside);
		if (Outside)
		{
			CRect ClientRect;
			GetClientRect(&ClientRect);
			if (ClientRect.PtInRect(point))
			{
				KillTimer(TID_SCROLLDOWN);
				KillTimer(TID_SCROLLUP);
				m_Interval = 0;
				Index = LB_ERR;
				Outside = FALSE;
			}
			else
			{
				DoTheScrolling(point, ClientRect);
			}
		}
		else
		{
			KillTimer(TID_SCROLLDOWN);
			KillTimer(TID_SCROLLUP);
			m_Interval = 0;
		}

		if (Index != m_MoveToIndex && !Outside)
		{
			DrawTheLines(Index);
		}
	}
}

void CDragAndDropListBox::OnTimer(UINT_PTR nIDEvent)
{
	// TODO: Add your message handler code here and/or call default

	if (nIDEvent == TID_SCROLLDOWN)
	{
		DrawTheLines(m_MoveToIndex+1);
		SetTopIndex(GetTopIndex()+1);
	}
	else if (nIDEvent == TID_SCROLLUP)
	{
		DrawTheLines(m_MoveToIndex-1);
		SetTopIndex(GetTopIndex()-1);
	}
	CListBox::OnTimer(nIDEvent);
}


void CDragAndDropListBox::DrawTheLines(int Index)
{
	CRect ClientRect;
	GetClientRect(&ClientRect);
	CDC *pDC = GetDC();

	CRect Rect;
	int Width = 2;
	if (m_MoveToIndex != m_MovingIndex)
	{
		CPen Pen(PS_SOLID, Width, RGB(255, 255, 255));
		CPen *pOldPen = pDC->SelectObject(&Pen);
		if (m_MoveToIndex != LB_ERR)
		{
			GetItemRect(m_MoveToIndex, &Rect);
			if (ClientRect.PtInRect(Rect.TopLeft()))
			{
				pDC->MoveTo(Rect.left, Rect.top+1);
				pDC->LineTo(Rect.right-(Width/2), Rect.top+1);
			}
		}
		else
		{
			GetItemRect(GetCount()-1, &Rect);
			if (ClientRect.PtInRect(CPoint(0, Rect.bottom)))
			{
				pDC->MoveTo(Rect.left, Rect.bottom);
				pDC->LineTo(Rect.right-(Width/2), Rect.bottom);
			}
		}
		pDC->SelectObject(pOldPen);
	}

	m_MoveToIndex = Index;
	if (m_MoveToIndex != m_MovingIndex)
	{
		CPen Pen(PS_SOLID, Width, RGB(0, 0, 0));
		CPen *pOldPen = pDC->SelectObject(&Pen);
		if (m_MoveToIndex != LB_ERR)
		{
			GetItemRect(Index, &Rect);
			if (ClientRect.PtInRect(Rect.TopLeft()))
			{
				pDC->MoveTo(Rect.left, Rect.top+1);
				pDC->LineTo(Rect.right-(Width/2), Rect.top+1);
			}
		}
		else
		{
			GetItemRect(GetCount()-1, &Rect);
			if (ClientRect.PtInRect(CPoint(0, Rect.bottom)))
			{
				pDC->MoveTo(Rect.left, Rect.bottom);
				pDC->LineTo(Rect.right-(Width/2), Rect.bottom);
			}
		}
		pDC->SelectObject(pOldPen);
	}

	ReleaseDC(pDC);
}

void CDragAndDropListBox::InsertDragedItem()
{
	CString Text;
	DWORD_PTR ItemData;
	GetText(m_MovingIndex, Text);
	ItemData = GetItemData(m_MovingIndex);
	if (m_MovingIndex < m_MoveToIndex)
	{
		SetRedraw(FALSE);
		int TopIndex = GetTopIndex();
		int NewIndex = InsertString(m_MoveToIndex, Text);
		SetItemData(NewIndex, ItemData);
		DeleteString(m_MovingIndex);
		SetCurSel(NewIndex-1);
		SetTopIndex(TopIndex);
		SetRedraw(TRUE);
		Invalidate();
		UpdateWindow();
	}

	else if (m_MovingIndex > m_MoveToIndex)
	{
		SetRedraw(FALSE);
		int TopIndex = GetTopIndex();
		DeleteString(m_MovingIndex);
		int NewIndex = InsertString(m_MoveToIndex, Text);
		SetItemData(NewIndex, ItemData);
		SetCurSel(NewIndex);
		SetTopIndex(TopIndex);
		SetRedraw(TRUE);
		Invalidate();
		UpdateWindow();
	}
	GetParent()->SendMessage(WM_COMMAND, MAKEWPARAM(GetDlgCtrlID(), LBN_SELCHANGE), (LPARAM)m_hWnd);
}


void CDragAndDropListBox::DoTheScrolling(CPoint Point , CRect ClientRect)
{
	if (Point.y > ClientRect.Height())
	{
		DWORD Interval = 250 /(1+(Point.y - ClientRect.Height()) /GetItemHeight(0));
		if (Interval != m_Interval)
		{
			m_Interval = Interval;
			SetTimer(TID_SCROLLDOWN, Interval, NULL);
			OnTimer(TID_SCROLLUP);
		}
	}
	else if (Point.y<0)
	{
		DWORD Interval = 150 / (1+(abs(Point.y)/GetItemHeight(1)));
		if (Interval != m_Interval)
		{
			m_Interval = Interval;
			SetTimer(TID_SCROLLDOWN, Interval, NULL);
			OnTimer(TID_SCROLLUP);
		}
	}
	else
	{
		KillTimer(TID_SCROLLDOWN);
		KillTimer(TID_SCROLLUP);
		m_Interval = 0;
	}
}

