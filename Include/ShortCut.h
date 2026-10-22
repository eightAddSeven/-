#ifndef SHORT_H
#define SHORT_H
#include <string>
using namespace std;
const int LENGTH = 64;
enum ShortcutType {TYPE_CMD, TYPE_MACRO, TYPE_USERCMD, TYPE_PLUGINCMD, TYPE_INVALID};

struct ShortCut
{
	char _name[LENGTH];
	bool _isCtrl;
	bool _isAit;
	bool _isShift;
	unsigned char _key;
	bool _canModefiyName;
	ShortCut():_isShift(false), _isCtrl(false), _isAit(false), _key(0), _canModefiyName(true)
	{
		_name[0] = '\0';
	};

	ShortCut(ShortCut &shorucut)
	{
		this->_isAit   = shorucut._isAit;
		this->_isCtrl  = shorucut._isCtrl;
		this->_isShift = shorucut._isShift;
		this->_key = shorucut._key;
		strcpy(this->_name, shorucut._name);
	};

	ShortCut(const char *name, bool isCtrl, bool isAlt, bool isShift, unsigned char key) :\
		_isCtrl(isCtrl), _isAit(isAlt), _isShift(isShift), _key(key)
	{
		_name[0] = '\0';
		if (name)
		{
			strcpy(_name, name);
		}
	};

	bool friend operator == (const ShortCut &a, const ShortCut &b) 
	{
		return ((strcmp(a._name, b._name) == 0) && (a._isCtrl == b._isCtrl) 
			&& (a._isAit== b._isAit) && (a._isShift == b._isShift) && (a._key == b._key));
	};

	bool friend operator != (const ShortCut &a, const ShortCut &b) 
	{
		return !((strcmp(a._name, b._name) == 0) && (a._isCtrl == b._isCtrl) 
			&& (a._isAit == b._isAit) && (a._isShift == b._isShift) && (a._key == b._key));
	};

	void copyShortcut(const ShortCut & sc) 
	{
		if (this != &sc)
		{
			strcpy(this->_name, sc._name);
			this->_isAit = sc._isAit;
			this->_isCtrl = sc._isCtrl;
			this->_isShift = sc._isShift;
			this->_key = sc._key;
			this->_canModefiyName = sc._canModefiyName;
		}
	};

	bool isValid() const 
	{
		if (_key == 0)
		{
			return false;
		}
		// the following keys are always valid (NUMPAD & F1~F12 + ESC + HOME + END)
		if (((_key >= 0x60) && (_key <= 0x69)) || ((_key >= 0x70) && (_key <= 0x7B)) || 
			(_key == 0x1B) || (_key == 0x24) || (_key == 0x23))
		{
			return true;
		}
		// the remain keys need at least Ctrl or Alt
		if ((_isCtrl) || (_isAit))
		{
			return true;
		}
		return false;
	};
	void setNameReadOnly(bool canBeModified = false) 
	{
		_canModefiyName = canBeModified;
	}

	//string toString() const;
	string toMenuItemString(int cmdID) 
	{
		string str = _name;
		if (cmdID)
		{
			//getNameStrFromCmd(cmdID, str);
		}
		str += "\t";
		//str += toString();
		return str;
	};
};


class PluginCmdShortcut : public ShortCut 
{
	friend class Parameters;
public:
	PluginCmdShortcut(ShortCut sc) : ShortCut(sc), _id(0), _internalID(-1) 
	{
		_moduleName[0] = '\0';
	};

	PluginCmdShortcut():_id(0), _internalID(0)
	{
		_moduleName[0] = '\0';
	};

	PluginCmdShortcut(const PluginCmdShortcut &ps)
	{
		this->_id = ps._id;
		this->_internalID = ps._internalID;
		strcpy(this->_moduleName, ps._moduleName);
	};

	PluginCmdShortcut(ShortCut sc, int cmdID) : ShortCut(sc), _id(cmdID), _internalID(-1) 
	{
		_moduleName[0] = '\0';
	};

	PluginCmdShortcut(ShortCut sc, int cmdID, const char *moduleName, unsigned short internalID) :\
		ShortCut(sc), _id(cmdID), _internalID(internalID)
	{
		strcpy(_moduleName, moduleName);
	};

	bool isValid() const 
	{
		if (!ShortCut::isValid())
		{
			return false;
		}
		if ((!_moduleName[0]) || (_internalID == -1))
		{
			return false;
		}
		return true;
	}
	const char * getModuleName() const 
	{
		return _moduleName;
	};

	int getInternalID() const 
	{
		return _internalID;
	};

	unsigned long getID() const 
	{
		return _id;
	};

protected :
	unsigned long _id;
	char _moduleName[LENGTH];
	int _internalID;
};

class Accelerator 
{
	friend class ShortcutMapper;
public:
	Accelerator():_hAccTable(NULL), _didCopy(false), _pAccelArray(NULL), _nbAccelItems(0)
	{
	};

	~Accelerator()
	{
		if (_didCopy)
		{
			::DestroyAcceleratorTable(_hAccTable);
		}
	};

	void init(HACCEL hAccel) 
	{
		_hAccTable = hAccel;
		_nbOriginalAccelItem = ::CopyAcceleratorTable(_hAccTable, NULL, 0);
	};

	HACCEL getAccTable() const {return _hAccTable;};

	bool uptdateShortcuts(HWND nppHandle = NULL);

	void coloneAccelTable() 
	{
		copyAccelArray();
		reNew();
	};


private:
	HACCEL _hAccTable;
	bool _didCopy;

	ACCEL *_pAccelArray;
	int _nbOriginalAccelItem;
	int _nbAccelItems;


	size_t copyAccelArray(int nbMacro2add = 0, int nbUserCmd2add = 0, int nbPluginCmd2add = 0) {
		int newSize = _nbOriginalAccelItem + nbMacro2add + nbUserCmd2add + nbPluginCmd2add;
		_nbAccelItems = newSize;

		if (_pAccelArray)
		{
			delete [] _pAccelArray;
		}
		_pAccelArray = new ACCEL[_nbAccelItems];

		::CopyAcceleratorTable(_hAccTable, _pAccelArray, _nbOriginalAccelItem);
		return newSize;
	};

	void reNew() 
	{
		if (!_didCopy)
		{
			_didCopy = true;
		}
		else
		{
			::DestroyAcceleratorTable(_hAccTable);
		}

		_hAccTable = ::CreateAcceleratorTable(_pAccelArray, _nbAccelItems);
	};
};

#endif
