#ifndef _UTIL_CPP_ALLOCATION_H_
#define _UTIL_CPP_ALLOCATION_H_

extern void* operator new  (size_t size);
extern void* operator new[](size_t size);
extern void  operator delete  (void* ptr);
extern void  operator delete[](void* ptr);
extern void  operator delete  (void* ptr, size_t size);
extern void  operator delete[](void* ptr, size_t size);

#endif
