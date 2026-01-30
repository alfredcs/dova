import apiClient from './client';
import type {
  UserProfile,
  ProfileUpdateRequest,
  RecommendationsResponse
} from './types';

export async function getProfile(): Promise<UserProfile> {
  const response = await apiClient.get<UserProfile>('/v1/profile');
  return response.data;
}

export async function updateProfile(data: ProfileUpdateRequest): Promise<UserProfile> {
  const response = await apiClient.put<UserProfile>('/v1/profile', data);
  return response.data;
}

export async function getRecommendations(): Promise<RecommendationsResponse> {
  const response = await apiClient.get<RecommendationsResponse>('/v1/profile/recommendations');
  return response.data;
}
