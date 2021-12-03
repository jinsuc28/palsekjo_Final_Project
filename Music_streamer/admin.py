from django.contrib import admin
from .models import UserSong
from import_export.admin import ExportActionModelAdmin, ImportExportMixin, ImportMixin




class UserSongAdmin(ImportExportMixin, admin.ModelAdmin):
    pass

admin.site.register(UserSong, UserSongAdmin)