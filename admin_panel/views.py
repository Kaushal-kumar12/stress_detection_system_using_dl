from django.shortcuts import redirect, render

def dashboard(request):
    return render(request, 'admin/admin_dashboard.html')  # Ensure the template path is correct

def admin_dashboard(request):
    if not request.user.is_staff:
        return redirect('login')

    return render(request, 'admin_panel/admin_dashboard.html')